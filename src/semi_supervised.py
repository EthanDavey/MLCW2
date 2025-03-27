import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset, Dataset
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import json
from datetime import datetime
import random
from copy import deepcopy
import torch.nn.functional as F

from clustering import load_features, perform_kmeans
from typicality import calculate_typicality, select_examples

def parse_args():
    parser = argparse.ArgumentParser(description='Semi-Supervised Active Learning Experiment with Self-Training and Consistency Regularization')
    parser.add_argument('--budget', type=int, default=10, 
                        help='Budget (B) - number of samples to select')
    parser.add_argument('--exp_dir', type=str, default='experiments/semi_supervised',
                        help='Directory to save experiment results')
    parser.add_argument('--epochs', type=int, default=10000,
                        help='Number of epochs to train the model')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--features_path', type=str, default='features/normalized_train_features.pkl',
                        help='Path to the feature file')
    parser.add_argument('--test_features_path', type=str, default='features/normalized_test_features.pkl',
                        help='Path to the test feature file')
    parser.add_argument('--consistency_weight', type=float, default=10.0,
                        help='Weight for consistency loss')
    parser.add_argument('--confidence_threshold', type=float, default=0.7,
                        help='Confidence threshold for using unlabeled examples')
    return parser.parse_args()

# Set random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    print(f"Random seed set to: {seed}")

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Constants
NUM_CLASSES = 10

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=256):
        super(MLPClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        return self.model(x)

class CIFAR10FeaturesDataset(Dataset):
    """Dataset that handles features extracted from CIFAR-10 images."""
    def __init__(self, features, labels, transform=None, target_transform=None):
        self.features = features
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        
        if self.transform:
            feature = self.transform(feature)
        if self.target_transform:
            label = self.target_transform(label)
            
        return feature, label

def create_data_loaders(train_features, train_labels, test_features, test_labels, labeled_indices, batch_size):
    """Create data loaders for labeled, unlabeled, and test data."""
    
    # Identify unlabeled indices
    unlabeled_indices = np.setdiff1d(np.arange(len(train_features)), labeled_indices)
    
    # Convert features to PyTorch tensors
    train_features_tensor = torch.FloatTensor(train_features)
    train_labels_tensor = torch.LongTensor(train_labels)
    test_features_tensor = torch.FloatTensor(test_features)
    test_labels_tensor = torch.LongTensor(test_labels)
    
    # Create labeled dataset
    labeled_dataset = TensorDataset(
        train_features_tensor[labeled_indices], 
        train_labels_tensor[labeled_indices]
    )
    labeled_loader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True)
    
    # Create unlabeled dataset (still includes labels for evaluation, but not used in training)
    unlabeled_dataset = TensorDataset(
        train_features_tensor[unlabeled_indices],
        train_labels_tensor[unlabeled_indices]
    )
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=True)
    
    # Create test dataset
    test_dataset = TensorDataset(test_features_tensor, test_labels_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return labeled_loader, unlabeled_loader, test_loader

def train_model(model, labeled_loader, unlabeled_loader, criterion, optimizer, scheduler, device, 
               epochs=200, consistency_weight=1.0, confidence_threshold=0.8):
    model.train()
    
    for epoch in range(epochs):
        running_sup_loss = 0.0
        running_unsup_loss = 0.0
        
        # Make sure unlabeled_loader can be iterated multiple times if it's smaller than labeled_loader
        unlabeled_iterator = iter(unlabeled_loader)
        
        for labeled_data in labeled_loader:
            # Get labeled data
            labeled_inputs, labels = labeled_data
            labeled_inputs, labels = labeled_inputs.to(device), labels.to(device)
            
            # Get unlabeled data (and restart iterator if needed)
            try:
                unlabeled_inputs, _ = next(unlabeled_iterator)
            except StopIteration:
                unlabeled_iterator = iter(unlabeled_loader)
                unlabeled_inputs, _ = next(unlabeled_iterator)
            
            unlabeled_inputs = unlabeled_inputs.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass for labeled data (supervised loss)
            outputs_labeled = model(labeled_inputs)
            sup_loss = criterion(outputs_labeled, labels)
            
            # Forward pass for unlabeled data (consistency regularization)
            with torch.no_grad():
                # Generate pseudo-labels using the model
                outputs_unlabeled = model(unlabeled_inputs)
                pseudo_labels = F.softmax(outputs_unlabeled, dim=1)
                max_probs, targets_u = torch.max(pseudo_labels, dim=1)
                mask = max_probs.ge(confidence_threshold).float()
            
            # Create multiple augmented versions of the unlabeled data
            noise_levels = [0.05, 0.1, 0.15]
            unsup_losses = []
            
            for noise in noise_levels:
                # Apply noise augmentation
                unlabeled_inputs_aug = unlabeled_inputs + noise * torch.randn_like(unlabeled_inputs).to(device)
                outputs_unlabeled_aug = model(unlabeled_inputs_aug)
                
                # Compute KL divergence between predictions and pseudo-labels
                log_probs = F.log_softmax(outputs_unlabeled_aug, dim=1)
                unsup_loss = F.kl_div(log_probs, pseudo_labels, reduction='none').sum(dim=1)
                unsup_losses.append(torch.mean(mask * unsup_loss))
            
            # Average the unsupervised loss across all augmentations
            unsup_loss = sum(unsup_losses) / len(unsup_losses)
            
            # Total loss
            loss = sup_loss + consistency_weight * unsup_loss
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            running_sup_loss += sup_loss.item()
            running_unsup_loss += unsup_loss.item() if unsup_loss.item() > 0 else 0
        
        # Adjust learning rate
        if scheduler is not None:
            scheduler.step()
        
        # Print statistics
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Sup Loss: {running_sup_loss/len(labeled_loader):.4f}, '
                  f'Unsup Loss: {running_unsup_loss/len(labeled_loader):.4f}, '
                  f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
    
    return model

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    return accuracy

def create_results_plot(exp_dir, tpc_accuracy, random_accuracy):
    """Create and save a visualization of results for a single experiment."""
    plt.figure(figsize=(8, 6))
    
    # Create a bar chart comparing TPC and Random
    methods = ['Random', 'TPC']
    accuracies = [random_accuracy, tpc_accuracy]
    colors = ['black', 'blue']
    
    # Plot the bars
    bars = plt.bar(methods, accuracies, color=colors, width=0.5)
    
    # Add accuracy values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.4f}', ha='center', va='bottom', fontsize=12)
    
    
    # Add labels and title
    plt.ylabel('Test Accuracy')
    plt.title('TPC vs Random Selection (Semi-Supervised)')
    plt.ylim(0, 1.0)  # Set y-axis from 0 to 1 for accuracy
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the plot
    plt.tight_layout()
    os.makedirs(os.path.join(exp_dir, 'plots'), exist_ok=True)
    plt.savefig(os.path.join(exp_dir, 'plots', 'results.png'))
    print(f"Results plot saved to {exp_dir}/plots/results.png")
    plt.close()

def save_experiment_config(exp_dir, args):
    """Save experiment configuration to a JSON file."""
    config = vars(args)
    config['device'] = str(device)
    config['datetime'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

def run_experiment(args, exp_dir, seed):
    """Run the experiment with the specified parameters."""
    # Set random seed
    set_seed(seed)
    
    print(f"Using device: {device}")
    
    # Create experiment directories
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'clustering'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'results'), exist_ok=True)
    
    # Load the features
    train_features, train_labels = load_features(args.features_path)
    test_features, test_labels = load_features(args.test_features_path)
    
    # ======= TPC Selection =======
    print("\n=== TPC Sample Selection ===")
    
    # Perform k-means clustering with B clusters
    print(f"Creating {args.budget} clusters...")
    cluster_labels, centroids, kmeans_model = perform_kmeans(train_features, args.budget, random_state=seed)
    
    clustering_results = {
        'n_clusters': int(args.budget),
        'cluster_labels': cluster_labels.tolist() if isinstance(cluster_labels, np.ndarray) else cluster_labels,
        'centroids': centroids.tolist() if isinstance(centroids, np.ndarray) else centroids
    }
    
    with open(os.path.join(exp_dir, 'clustering', 'clusters.json'), 'w') as f:
        json.dump(clustering_results, f, indent=4)
    
    # Calculate typicality scores
    typicality_scores = calculate_typicality(train_features)
    
    # Select samples using TPC strategy (one from each cluster)
    tpc_indices = select_examples(
        cluster_labels, 
        typicality_scores, 
        budget=args.budget, 
        labeled_indices=np.array([], dtype=int)
    )
    
    print(f"Selected {len(tpc_indices)} examples using TPC")
    
    # ======= Random Selection =======
    print("\n=== Random Sample Selection ===")
    # Select random samples
    random_indices = np.random.choice(len(train_features), args.budget, replace=False)
    print(f"Selected {len(random_indices)} examples randomly")
    
    # Save sampling info
    sampling_results = {
        'budget': int(args.budget),
        'tpc_indices': tpc_indices.tolist(),
        'random_indices': random_indices.tolist()
    }
    
    with open(os.path.join(exp_dir, 'results', 'sampling.json'), 'w') as f:
        json.dump(sampling_results, f, indent=4)
    
    # ======= Model Training & Evaluation =======
    # Create data loaders for both strategies
    tpc_labeled_loader, tpc_unlabeled_loader, test_loader = create_data_loaders(
        train_features, train_labels, 
        test_features, test_labels, 
        tpc_indices, args.batch_size
    )
    
    random_labeled_loader, random_unlabeled_loader, _ = create_data_loaders(
        train_features, train_labels, 
        test_features, test_labels, 
        random_indices, args.batch_size
    )
    
    # Train and evaluate TPC model with semi-supervised learning
    print("\n=== Training TPC Model (Semi-Supervised) ===")
    tpc_model = MLPClassifier(train_features.shape[1], NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(tpc_model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    tpc_model = train_model(
        tpc_model, tpc_labeled_loader, tpc_unlabeled_loader, 
        criterion, optimizer, scheduler, device, 
        epochs=args.epochs, 
        consistency_weight=args.consistency_weight,
        confidence_threshold=args.confidence_threshold
    )
    
    tpc_accuracy = evaluate_model(tpc_model, test_loader, device)
    print(f"TPC Model Accuracy: {tpc_accuracy:.4f}")
    
    # Train and evaluate Random model with semi-supervised learning
    print("\n=== Training Random Model (Semi-Supervised) ===")
    random_model = MLPClassifier(train_features.shape[1], NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(random_model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    random_model = train_model(
        random_model, random_labeled_loader, random_unlabeled_loader, 
        criterion, optimizer, scheduler, device, 
        epochs=args.epochs,
        consistency_weight=args.consistency_weight,
        confidence_threshold=args.confidence_threshold
    )
    
    random_accuracy = evaluate_model(random_model, test_loader, device)
    print(f"Random Model Accuracy: {random_accuracy:.4f}")
    
    # Save performance results
    performance_results = {
        'budget': int(args.budget),
        'tpc_accuracy': float(tpc_accuracy),
        'random_accuracy': float(random_accuracy),
        'accuracy_diff': float(tpc_accuracy - random_accuracy),
        'improvement_percentage': float((tpc_accuracy - random_accuracy) / random_accuracy * 100) if random_accuracy > 0 else 0
    }
    
    with open(os.path.join(exp_dir, 'results', 'performance.json'), 'w') as f:
        json.dump(performance_results, f, indent=4)
    
    # Create and save visualization of results
    create_results_plot(exp_dir, tpc_accuracy, random_accuracy)
    
    print(f"Results saved to {exp_dir}/results/")
    
    # Return results
    return performance_results

def main():
    # Parse arguments
    args = parse_args()
    
    print("\n" + "="*80)
    print(f"Starting Semi-Supervised Learning Experiment")
    print("="*80 + "\n")
    
    # Create experiment directory
    exp_dir = args.exp_dir
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save configuration
    save_experiment_config(exp_dir, args)
    
    # Run experiment
    results = run_experiment(args, exp_dir, args.seed)
    
    # Print summary
    print("\n" + "="*80)
    print("Experiment Summary:")
    print(f"TPC Accuracy: {results['tpc_accuracy']:.4f}")
    print(f"Random Accuracy: {results['random_accuracy']:.4f}")
    print(f"Improvement: {results['accuracy_diff']:.4f} ({results['improvement_percentage']:.2f}%)")
    print("="*80 + "\n")
    
    print("\nExperiment completed successfully!")

if __name__ == "__main__":
    main() 
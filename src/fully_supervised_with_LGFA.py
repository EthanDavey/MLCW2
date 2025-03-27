import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import json
import copy

from clustering import load_features, perform_kmeans
from typicality import calculate_typicality, select_examples

def parse_args():
    parser = argparse.ArgumentParser(description='Fully Supervised with Label Guided Feature Adaptation (LGFA) Active Learning Experiment')
    parser.add_argument('--budget', type=int, default=10, 
                        help='Budget (B) - number of samples to select at each iteration')
    parser.add_argument('--iterations', type=int, default=6, 
                        help='Number of iterations to run (each adds B samples)')
    parser.add_argument('--repetitions', type=int, default=1,
                        help='Number of times to repeat the experiment with different random seeds')
    parser.add_argument('--exp_dir', type=str, default='experiments/fully_supervised_with_LGFA_B10',
                        help='Directory to save experiment results')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs to train the model in each iteration')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (will be incremented for each repetition)')
    parser.add_argument('--features_path', type=str, default='features/normalized_train_features.pkl',
                        help='Path to the feature file for training data')
    parser.add_argument('--test_features_path', type=str, default='features/normalized_test_features.pkl',
                        help='Path to the feature file for test data')
    parser.add_argument('--consistency_weight', type=float, default=0.5,
                        help='Weight for consistency loss')
    parser.add_argument('--separation_weight', type=float, default=0.5,
                        help='Weight for separation loss')
    parser.add_argument('--original_sse_results', type=str, default='experiments/fully_supervised_with_SSE_B10/rep_0/results/cumulative_results.json',
                        help='Path to original SSE results for comparison')
    return parser.parse_args()

# Set random seed for reproducibility
def set_seed(seed):
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

# Feature Adaptation Network
class FeatureAdaptationNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=512):
        super(FeatureAdaptationNetwork, self).__init__()
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.BatchNorm1d(input_dim)
        )
        self.original_input_dim = input_dim
        
        # Initialize weights to be close to identity transformation
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        adapted_features = self.adapter(x)
        # Normalize the features to avoid NaN issues
        norm = torch.norm(adapted_features, p=2, dim=1, keepdim=True) + 1e-8
        adapted_features = adapted_features / norm
        return adapted_features

# Simple linear classifier that uses adapted features
class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        return self.linear(x)

# Combined model that includes both feature adaptation and classification
class LGFAModel(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=512):
        super(LGFAModel, self).__init__()
        self.feature_adapter = FeatureAdaptationNetwork(input_dim, hidden_dim)
        self.classifier = LinearClassifier(input_dim, num_classes)
        
    def forward(self, x, return_adapted=False):
        adapted_features = self.feature_adapter(x)
        outputs = self.classifier(adapted_features)
        if return_adapted:
            return outputs, adapted_features
        return outputs
    
    def get_adapted_features(self, x):
        with torch.no_grad():
            adapted_features = self.feature_adapter(x)
        return adapted_features

# Calculate inter-class separation loss
def calculate_separation_loss(features, labels):
    # Get unique class labels
    unique_labels = torch.unique(labels)
    num_classes = len(unique_labels)
    
    # Calculate class means
    class_means = []
    for class_idx in unique_labels:
        class_mask = (labels == class_idx)
        if torch.sum(class_mask) > 0:  # Check if we have examples of this class
            class_features = features[class_mask]
            class_mean = torch.mean(class_features, dim=0)
            class_means.append(class_mean)
    
    # If we don't have at least 2 classes, return zero loss
    if len(class_means) < 2:
        return torch.tensor(0.0).to(features.device)
    
    # Calculate pairwise distances between class means
    class_means = torch.stack(class_means)
    
    # Compute the mean of all pairwise distances (higher is better for separation)
    pairwise_distances = torch.cdist(class_means, class_means, p=2)
    
    # Create a mask to ignore the diagonal (distances to self)
    mask = ~torch.eye(num_classes, dtype=torch.bool, device=features.device)
    valid_distances = pairwise_distances[mask]
    
    # Separation loss is negative of the mean distance (to maximize distance)
    if len(valid_distances) > 0:
        separation_loss = -torch.mean(valid_distances)
    else:
        separation_loss = torch.tensor(0.0).to(features.device)
    
    return separation_loss

def train_lgfa_model(model, original_features, train_loader, criterion, optimizer, scheduler, device, 
                   epochs=200, consistency_weight=0.5, separation_weight=0.5):
    model.train()
    
    # Initialize learning rate warmup
    warmup_epochs = min(20, epochs // 5)
    
    for epoch in range(epochs):
        train_loss = 0.0
        classification_loss_total = 0.0
        consistency_loss_total = 0.0
        separation_loss_total = 0.0
        
        # Adjust losses based on epoch
        if epoch < warmup_epochs:
            # During warmup, gradually increase the weights
            current_consistency_weight = consistency_weight * (epoch + 1) / warmup_epochs
            current_separation_weight = separation_weight * (epoch + 1) / warmup_epochs
        else:
            current_consistency_weight = consistency_weight
            current_separation_weight = separation_weight
        
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs, adapted_features = model(features, return_adapted=True)
            
            # 1. Classification loss (cross-entropy)
            classification_loss = criterion(outputs, labels)
            
            # 2. Consistency loss (MSE between original and adapted features)
            # Scale down to avoid numerical instability
            consistency_loss = F.mse_loss(adapted_features, features) * 0.1
            
            # 3. Separation loss (maximize inter-class distance)
            separation_loss = calculate_separation_loss(adapted_features, labels) * 0.1
            
            # Check for NaN losses and avoid them
            if torch.isnan(classification_loss) or torch.isinf(classification_loss):
                classification_loss = torch.tensor(0.0).to(device)
            
            if torch.isnan(consistency_loss) or torch.isinf(consistency_loss):
                consistency_loss = torch.tensor(0.0).to(device)
                
            if torch.isnan(separation_loss) or torch.isinf(separation_loss):
                separation_loss = torch.tensor(0.0).to(device)
            
            # Combine all losses
            total_loss = classification_loss + current_consistency_weight * consistency_loss + current_separation_weight * separation_loss
            
            # Backward and optimize
            if not torch.isnan(total_loss) and not torch.isinf(total_loss):
                total_loss.backward()
                # Apply gradient clipping to avoid exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            train_loss += total_loss.item()
            classification_loss_total += classification_loss.item()
            consistency_loss_total += consistency_loss.item() if not torch.isnan(consistency_loss) else 0
            separation_loss_total += separation_loss.item() if not torch.isnan(separation_loss) else 0
        
        # Adjust learning rate
        scheduler.step()
        
        # Print statistics
        if (epoch + 1) % 20 == 0:  # Print every 20 epochs
            print(f'Epoch [{epoch+1}/{epochs}], Total Loss: {train_loss/len(train_loader):.4f}, '
                  f'Class Loss: {classification_loss_total/len(train_loader):.4f}, '
                  f'Consist Loss: {consistency_loss_total/len(train_loader):.4f}, '
                  f'Sep Loss: {separation_loss_total/len(train_loader):.4f}, '
                  f'LR: {scheduler.get_last_lr()[0]:.6f}')
    
    return model

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    return accuracy

def get_feature_loaders(train_features, train_labels, test_features, test_labels, labeled_indices, batch_size):
    """Create data loaders for features directly, using only labeled examples for training."""
    # Create tensor datasets
    if labeled_indices is not None:
        # Select only labeled features and labels
        train_features_subset = train_features[labeled_indices]
        train_labels_subset = train_labels[labeled_indices]
        train_dataset = TensorDataset(torch.FloatTensor(train_features_subset), torch.LongTensor(train_labels_subset))
    else:
        train_dataset = TensorDataset(torch.FloatTensor(train_features), torch.LongTensor(train_labels))
    
    test_dataset = TensorDataset(torch.FloatTensor(test_features), torch.LongTensor(test_labels))
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def initialize_plot(exp_dir):
    plot_dir = os.path.join(exp_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.title('LGFA vs SSE vs Random (Fully Supervised with Feature Adaptation)')
    plt.xlabel('Cumulative Budget (Number of Labeled Samples)')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    # Create empty lists for storing results
    budgets = []
    lgfa_accuracies = []
    random_accuracies = []
    
    return budgets, lgfa_accuracies, random_accuracies

def update_plot(exp_dir, budgets, lgfa_accuracies, random_accuracies, original_sse_results, iteration):
    plot_dir = os.path.join(exp_dir, 'plots')
    
    plt.clf()  # Clear the current figure
    plt.figure(figsize=(10, 6))
    
    # Get original SSE results for comparison
    with open(original_sse_results, 'r') as f:
        sse_data = json.load(f)
    
    sse_budgets = sse_data['budgets']
    sse_tpc_accuracies = sse_data['tpc_accuracies']
    sse_random_accuracies = sse_data['random_accuracies']
    
    # Plot up to the current iteration
    current_budgets = budgets[:iteration+1]
    current_lgfa = lgfa_accuracies[:iteration+1]
    current_random = random_accuracies[:iteration+1]
    current_sse_tpc = sse_tpc_accuracies[:iteration+1]
    current_sse_random = sse_random_accuracies[:iteration+1]
    
    # Add markers to the plot points with larger marker size
    plt.plot(current_budgets, current_lgfa, 'r-o', label='LGFA', markersize=6, markerfacecolor='red', markeredgecolor='red')
    plt.plot(current_budgets, current_sse_tpc, 'b-o', label='SSE-TPC', markersize=6, markerfacecolor='blue', markeredgecolor='blue')
    plt.plot(current_budgets, current_random, 'k-o', label='LGFA-Random', markersize=6, markerfacecolor='black', markeredgecolor='black')
    plt.plot(current_budgets, current_sse_random, 'k--o', label='SSE-Random', markersize=6, markerfacecolor='black', markeredgecolor='black')
    
    plt.title('LGFA vs SSE vs Random (Fully Supervised with Feature Adaptation)')
    plt.xlabel('Cumulative Budget (Number of Labeled Samples)')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    
    # Save the plot with iteration number
    plt.savefig(os.path.join(plot_dir, f'accuracy_comparison_iter_{iteration}.png'))
    # Also save the latest version without iteration number for easy reference
    plt.savefig(os.path.join(plot_dir, 'accuracy_comparison_latest.png'))
    print(f'Plot saved to {plot_dir}/accuracy_comparison_iter_{iteration}.png')

def create_comparison_plot(exp_dir, lgfa_results, original_sse_results_path):
    """Create a final comparison plot between LGFA and SSE methods."""
    print("\n--- Creating Comparison Results Plot ---")
    
    # Load original SSE results
    with open(original_sse_results_path, 'r') as f:
        sse_data = json.load(f)
    
    sse_budgets = sse_data['budgets']
    sse_tpc_accuracies = sse_data['tpc_accuracies']
    sse_random_accuracies = sse_data['random_accuracies']
    
    # Get LGFA results
    lgfa_budgets = lgfa_results['budgets']
    lgfa_tpc_accuracies = lgfa_results['lgfa_accuracies']
    lgfa_random_accuracies = lgfa_results['random_accuracies']
    
    # Create the comparison plot
    plt.clf()
    plt.figure(figsize=(12, 8))
    
    # Plot lines with markers
    plt.plot(lgfa_budgets, lgfa_tpc_accuracies, 'r-o', label='LGFA-TPC', linewidth=2, markersize=8, markerfacecolor='red', markeredgecolor='red')
    plt.plot(sse_budgets, sse_tpc_accuracies, 'b-o', label='SSE-TPC', linewidth=2, markersize=8, markerfacecolor='blue', markeredgecolor='blue')
    plt.plot(lgfa_budgets, lgfa_random_accuracies, 'r--o', label='LGFA-Random', linewidth=2, markersize=8, markerfacecolor='red', markeredgecolor='red')
    plt.plot(sse_budgets, sse_random_accuracies, 'b--o', label='SSE-Random', linewidth=2, markersize=8, markerfacecolor='blue', markeredgecolor='blue')
    
    plt.title('LGFA vs SSE Comparison (Fully Supervised with Feature Adaptation)', fontsize=14)
    plt.xlabel('Cumulative Budget (Number of Labeled Samples)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Add improvement annotations
    for i, budget in enumerate(lgfa_budgets):
        improvement = lgfa_tpc_accuracies[i] - sse_tpc_accuracies[i]
        if improvement > 0:
            plt.annotate(f'+{improvement:.3f}', 
                         xy=(budget, lgfa_tpc_accuracies[i]), 
                         xytext=(budget, lgfa_tpc_accuracies[i] + 0.025),
                         arrowprops=dict(arrowstyle='->', color='green'),
                         ha='center', va='bottom', color='green', fontsize=9)
    
    # Save the comparison plot
    comparison_path = os.path.join(exp_dir, 'lgfa_vs_sse_comparison.png')
    plt.savefig(comparison_path)
    print(f"Comparison plot saved to {comparison_path}")
    
    # Create a table with improvement statistics
    improvement_data = {
        'budget': lgfa_budgets,
        'lgfa_tpc': lgfa_tpc_accuracies,
        'sse_tpc': sse_tpc_accuracies,
        'improvement': [lgfa_tpc_accuracies[i] - sse_tpc_accuracies[i] for i in range(len(lgfa_budgets))],
        'improvement_percentage': [(lgfa_tpc_accuracies[i] - sse_tpc_accuracies[i]) / sse_tpc_accuracies[i] * 100 for i in range(len(lgfa_budgets))],
        'lgfa_random': lgfa_random_accuracies,
        'sse_random': sse_random_accuracies
    }
    
    with open(os.path.join(exp_dir, 'comparison_results.json'), 'w') as f:
        json.dump(improvement_data, f, indent=4)
    
    print(f"Comparison statistics saved to {exp_dir}/comparison_results.json")
    
    return improvement_data

def run_single_experiment(args, exp_dir, seed):
    """Run a single experiment with the specified parameters."""
    # Set random seed for this repetition
    set_seed(seed)
    
    print(f"Using device: {device}")
    
    # Create experiment directories
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'clustering'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'results'), exist_ok=True)
    
    # Load the features from the pre-trained self-supervised model
    train_features, train_labels = load_features(args.features_path)
    test_features, test_labels = load_features(args.test_features_path)
    
    # Get feature dimensions
    feature_dim = train_features.shape[1]
    
    # Initialize empty arrays for labeled indices
    lgfa_labeled_indices = np.array([], dtype=int)
    random_labeled_indices = np.array([], dtype=int)
    
    # Initialize the plot
    budgets, lgfa_accuracies, random_accuracies = initialize_plot(exp_dir)
    
    # Create tensor datasets for all features (for feature adaptation)
    all_train_tensor = torch.FloatTensor(train_features)
    all_test_tensor = torch.FloatTensor(test_features)
    
    # Initialize feature adaptation network (will be updated at each iteration)
    feature_adapter = FeatureAdaptationNetwork(feature_dim).to(device)
    
    # Main active learning loop
    current_budget = 0
    
    # Store results for comparison
    results = {
        'budgets': [],
        'lgfa_accuracies': [],
        'random_accuracies': [],
        'lgfa_labeled_indices': [],
        'random_labeled_indices': []
    }
    
    for iteration in range(args.iterations):
        print(f"\n=== Iteration {iteration} (Current Budget: {current_budget}) ===")
        
        # Convert train features to tensor for adaptation
        train_tensor = torch.FloatTensor(train_features).to(device)
        
        # Update features if we have labeled examples
        if iteration > 0:
            print("Adapting features based on labeled examples...")
            with torch.no_grad():
                adapted_features = feature_adapter(train_tensor).cpu().numpy()
                
                # Check for NaN or Infinity and replace with original features if found
                if np.isnan(adapted_features).any() or np.isinf(adapted_features).any():
                    print("Warning: NaN or Inf values detected in adapted features. Using original features instead.")
                    adapted_train_features = train_features
                else:
                    adapted_train_features = adapted_features
        else:
            # First iteration - use original features
            adapted_train_features = train_features
        
        # Perform k-means clustering on adapted features
        num_clusters = args.budget  # Number of clusters = budget
        print(f"Performing k-means clustering with {num_clusters} clusters...")
        cluster_labels, centroids, kmeans_model = perform_kmeans(adapted_train_features, num_clusters, random_state=seed+iteration)
        
        # Save clustering results
        clustering_results = {
            'iteration': int(iteration),
            'n_clusters': int(num_clusters),
            'cluster_labels': cluster_labels.tolist(),
        }
        
        with open(os.path.join(exp_dir, 'clustering', f'clusters_iter_{iteration}.json'), 'w') as f:
            json.dump(clustering_results, f, indent=4)
        
        # Calculate typicality scores on adapted features
        typicality_scores = calculate_typicality(adapted_train_features)
        
        # Select new examples using TPC strategy (one from each cluster)
        new_lgfa_indices = select_examples(
            cluster_labels, 
            typicality_scores, 
            budget=args.budget, 
            labeled_indices=lgfa_labeled_indices
        )
        
        # Select new examples using Random strategy
        # Exclude already labeled examples
        unlabeled_random_indices = np.setdiff1d(np.arange(len(train_features)), random_labeled_indices)
        new_random_indices = np.random.choice(unlabeled_random_indices, args.budget, replace=False)
        
        # Update labeled indices
        lgfa_labeled_indices = np.concatenate([lgfa_labeled_indices, new_lgfa_indices])
        random_labeled_indices = np.concatenate([random_labeled_indices, new_random_indices])
        
        # Save sampling information
        sampling_results = {
            'iteration': int(iteration),
            'lgfa_new_indices': new_lgfa_indices.tolist(),
            'random_new_indices': new_random_indices.tolist(),
            'lgfa_all_indices': lgfa_labeled_indices.tolist(),
            'random_all_indices': random_labeled_indices.tolist()
        }
        
        with open(os.path.join(exp_dir, 'results', f'sampling_iter_{iteration}.json'), 'w') as f:
            json.dump(sampling_results, f, indent=4)
        
        # Update current budget
        current_budget += args.budget
        
        # Create feature data loaders for both strategies
        lgfa_train_loader, test_loader = get_feature_loaders(
            train_features, train_labels, 
            test_features, test_labels, 
            lgfa_labeled_indices, args.batch_size
        )
        
        random_train_loader, _ = get_feature_loaders(
            train_features, train_labels, 
            test_features, test_labels, 
            random_labeled_indices, args.batch_size
        )
        
        # Train and evaluate LGFA model
        print("Training LGFA model...")
        lgfa_model = LGFAModel(feature_dim, NUM_CLASSES).to(device)
        criterion = nn.CrossEntropyLoss()
        # Reduce the learning rate to improve stability
        optimizer = optim.SGD(lgfa_model.parameters(), lr=0.5, momentum=0.9, nesterov=True, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        
        lgfa_model = train_lgfa_model(
            lgfa_model, 
            train_features,
            lgfa_train_loader, 
            criterion, 
            optimizer, 
            scheduler, 
            device, 
            epochs=args.epochs,
            consistency_weight=args.consistency_weight,
            separation_weight=args.separation_weight
        )
        
        lgfa_accuracy = evaluate_model(lgfa_model, test_loader, device)
        print(f"LGFA Model Accuracy: {lgfa_accuracy:.4f}")
        
        # Train and evaluate Random model with the same LGFA approach
        print("Training Random model with LGFA...")
        random_model = LGFAModel(feature_dim, NUM_CLASSES).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(random_model.parameters(), lr=0.5, momentum=0.9, nesterov=True, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        
        random_model = train_lgfa_model(
            random_model, 
            train_features,
            random_train_loader, 
            criterion, 
            optimizer, 
            scheduler, 
            device, 
            epochs=args.epochs,
            consistency_weight=args.consistency_weight,
            separation_weight=args.separation_weight
        )
        
        random_accuracy = evaluate_model(random_model, test_loader, device)
        print(f"Random Model with LGFA Accuracy: {random_accuracy:.4f}")
        
        # Save performance results
        performance_results = {
            'iteration': int(iteration),
            'budget': int(current_budget),
            'lgfa_accuracy': float(lgfa_accuracy),
            'random_accuracy': float(random_accuracy),
            'accuracy_diff': float(lgfa_accuracy - random_accuracy)
        }
        
        with open(os.path.join(exp_dir, 'results', f'performance_iter_{iteration}.json'), 'w') as f:
            json.dump(performance_results, f, indent=4)
        
        # Update feature adapter for the next iteration
        feature_adapter = copy.deepcopy(lgfa_model.feature_adapter)
        
        # Update plot data
        budgets.append(current_budget)
        lgfa_accuracies.append(lgfa_accuracy)
        random_accuracies.append(random_accuracy)
        
        # Update plot
        update_plot(exp_dir, budgets, lgfa_accuracies, random_accuracies, args.original_sse_results, iteration)
        
        # Update results for return
        results['budgets'].append(current_budget)
        results['lgfa_accuracies'].append(lgfa_accuracy)
        results['random_accuracies'].append(random_accuracy)
        results['lgfa_labeled_indices'] = lgfa_labeled_indices.tolist()
        results['random_labeled_indices'] = random_labeled_indices.tolist()
    
    # Save cumulative results
    with open(os.path.join(exp_dir, 'results', 'cumulative_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    # Create comparison plot with original SSE results
    comparison_results = create_comparison_plot(exp_dir, results, args.original_sse_results)
    
    return results

def main():
    # Parse arguments
    args = parse_args()
    
    # Create experiment directory
    exp_dir = args.exp_dir
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        config = vars(args)
        config['device'] = str(device)
        json.dump(config, f, indent=4)
    
    # Set seed for the first experiment
    rep_seed = args.seed
    
    # Run the experiment
    print(f"\n\n{'='*80}")
    print(f"Starting LGFA Experiment")
    print(f"{'='*80}\n")
    
    results = run_single_experiment(args, exp_dir, rep_seed)
    
    print("\nExperiment completed successfully!")
    print(f"Final LGFA accuracy: {results['lgfa_accuracies'][-1]:.4f}")
    print(f"Final Random accuracy: {results['random_accuracies'][-1]:.4f}")
    
    # Print comparison to original SSE
    with open(args.original_sse_results, 'r') as f:
        sse_data = json.load(f)
    
    final_lgfa_acc = results['lgfa_accuracies'][-1]
    final_sse_acc = sse_data['tpc_accuracies'][-1]
    improvement = final_lgfa_acc - final_sse_acc
    improvement_pct = (improvement / final_sse_acc) * 100
    
    print(f"\nComparison to original SSE:")
    print(f"Final SSE-TPC accuracy: {final_sse_acc:.4f}")
    print(f"LGFA improvement: {improvement:.4f} ({improvement_pct:.2f}%)")

if __name__ == "__main__":
    main() 
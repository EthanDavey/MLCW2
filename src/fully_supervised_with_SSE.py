import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import json

from clustering import load_features, perform_kmeans
from typicality import calculate_typicality, select_examples

def parse_args():
    parser = argparse.ArgumentParser(description='Fully Supervised with Self-Supervised Embeddings Active Learning Experiment')
    parser.add_argument('--budget', type=int, default=10, 
                        help='Budget (B) - number of samples to select at each iteration')
    parser.add_argument('--iterations', type=int, default=6, 
                        help='Number of iterations to run (each adds B samples)')
    parser.add_argument('--repetitions', type=int, default=1,
                        help='Number of times to repeat the experiment with different random seeds')
    parser.add_argument('--exp_dir', type=str, default='experiments/fully_supervised_with_SSE_B10',
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

# Simple linear classifier
class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        return self.linear(x)

def train_model(model, train_loader, criterion, optimizer, scheduler, device, epochs=200):
    model.train()
    
    for epoch in range(epochs):
        train_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Adjust learning rate
        scheduler.step()
        
        # Print statistics
        if (epoch + 1) % 20 == 0:  # Print every 20 epochs since we're using more epochs
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {train_loss/len(train_loader):.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')
    
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
    plt.title('TPC vs Random Selection (Fully Supervised with Self-Supervised Embeddings)')
    plt.xlabel('Cumulative Budget (Number of Labeled Samples)')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    # Create empty lists for storing results
    budgets = []
    tpc_accuracies = []
    random_accuracies = []
    
    return budgets, tpc_accuracies, random_accuracies

def update_plot(exp_dir, budgets, tpc_accuracies, random_accuracies, iteration):
    plot_dir = os.path.join(exp_dir, 'plots')
    
    plt.clf()  # Clear the current figure
    plt.figure(figsize=(10, 6))
    
    # Add markers to the plot points with larger marker size
    plt.plot(budgets, tpc_accuracies, 'b-o', label='TPC', markersize=6, markerfacecolor='blue', markeredgecolor='blue')
    plt.plot(budgets, random_accuracies, 'k-o', label='Random', markersize=6, markerfacecolor='black', markeredgecolor='black')
    
    plt.title('TPC vs Random Selection (Fully Supervised with Self-Supervised Embeddings)')
    plt.xlabel('Cumulative Budget (Number of Labeled Samples)')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    
    # Save the plot with iteration number
    plt.savefig(os.path.join(plot_dir, f'accuracy_comparison_iter_{iteration}.png'))
    # Also save the latest version without iteration number for easy reference
    plt.savefig(os.path.join(plot_dir, 'accuracy_comparison_latest.png'))
    print(f'Plot saved to {plot_dir}/accuracy_comparison_iter_{iteration}.png')

def create_aggregated_plot(parent_exp_dir, repetitions, iterations, budget):
    """Create an aggregated plot that shows the average performance across all repetitions with std dev shading."""
    print("\n--- Creating Aggregated Results ---")
    
    # Initialize lists to store data from all repetitions
    all_tpc_accuracies = []
    all_random_accuracies = []
    
    # Load data from each repetition
    for rep in range(repetitions):
        rep_dir = os.path.join(parent_exp_dir, f'rep_{rep}')
        with open(os.path.join(rep_dir, 'results', 'cumulative_results.json'), 'r') as f:
            results = json.load(f)
            
        all_tpc_accuracies.append(results['tpc_accuracies'])
        all_random_accuracies.append(results['random_accuracies'])
    
    # Convert to numpy arrays for easy calculation
    all_tpc_accuracies = np.array(all_tpc_accuracies)
    all_random_accuracies = np.array(all_random_accuracies)
    
    # Calculate mean and std dev
    mean_tpc = np.mean(all_tpc_accuracies, axis=0)
    std_tpc = np.std(all_tpc_accuracies, axis=0)
    
    mean_random = np.mean(all_random_accuracies, axis=0)
    std_random = np.std(all_random_accuracies, axis=0)
    
    # Create budget values
    budgets = [(i+1) * budget for i in range(iterations)]
    
    # Create the plot
    plt.clf()
    plt.figure(figsize=(10, 6))
    
    # Plot lines with markers
    plt.plot(budgets, mean_tpc, 'b-o', label='TPC', markersize=6, markerfacecolor='blue', markeredgecolor='blue')
    plt.plot(budgets, mean_random, 'k-o', label='Random', markersize=6, markerfacecolor='black', markeredgecolor='black')
    
    # Add shaded areas for standard deviation
    plt.fill_between(budgets, mean_tpc - std_tpc, mean_tpc + std_tpc, color='blue', alpha=0.2)
    plt.fill_between(budgets, mean_random - std_random, mean_random + std_random, color='black', alpha=0.2)
    
    plt.title('TPC vs Random Selection (Averaged Over Repetitions)')
    plt.xlabel('Cumulative Budget (Number of Labeled Samples)')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    
    # Save the aggregated plot
    plt.savefig(os.path.join(parent_exp_dir, 'accuracy_comparison_aggregated.png'))
    print(f"Aggregated plot saved to {parent_exp_dir}/accuracy_comparison_aggregated.png")
    
    # Save aggregated results to JSON
    aggregated_results = {
        'budgets': budgets,
        'tpc_mean': mean_tpc.tolist(),
        'tpc_std': std_tpc.tolist(),
        'random_mean': mean_random.tolist(),
        'random_std': std_random.tolist(),
        'repetitions': repetitions,
        'iterations': iterations,
        'budget_per_iteration': budget
    }
    
    with open(os.path.join(parent_exp_dir, 'aggregated_results.json'), 'w') as f:
        json.dump(aggregated_results, f, indent=4)
    
    print(f"Aggregated results saved to {parent_exp_dir}/aggregated_results.json")
    
    return aggregated_results


def run_single_experiment(args, rep_dir, seed):
    """Run a single experiment with the specified parameters."""
    # Set random seed for this repetition
    set_seed(seed)
    
    print(f"Using device: {device}")
    
    # Create experiment directories
    os.makedirs(rep_dir, exist_ok=True)
    os.makedirs(os.path.join(rep_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(rep_dir, 'clustering'), exist_ok=True)
    os.makedirs(os.path.join(rep_dir, 'results'), exist_ok=True)

    
    # Load the features from the pre-trained self-supervised model
    train_features, train_labels = load_features(args.features_path)
    test_features, test_labels = load_features(args.test_features_path)
    
    # Get feature dimensions
    feature_dim = train_features.shape[1]
    
    # Initialize empty arrays for labeled indices
    tpc_labeled_indices = np.array([], dtype=int)
    random_labeled_indices = np.array([], dtype=int)
    
    # Initialize the plot
    budgets, tpc_accuracies, random_accuracies = initialize_plot(rep_dir)
    
    # Main active learning loop
    current_budget = 0
    
    # Iterate for a fixed number of iterations
    for iteration in range(args.iterations):
        print(f"\n--- Starting Iteration {iteration} ---")
        
        # Calculate how many clusters to create for this iteration
        n_clusters = len(tpc_labeled_indices) + args.budget
        
        if n_clusters == 0:  # First iteration
            n_clusters = args.budget
        
        print(f"Creating {n_clusters} clusters...")
        
        # Perform k-means clustering
        cluster_labels, centroids, kmeans_model = perform_kmeans(train_features, n_clusters, random_state=seed)
        
        clustering_results = {
            'iteration': int(iteration),
            'n_clusters': int(n_clusters),
            'cluster_labels': cluster_labels.tolist() if isinstance(cluster_labels, np.ndarray) else cluster_labels,
            'centroids': centroids.tolist() if isinstance(centroids, np.ndarray) else centroids
        }
        
        with open(os.path.join(rep_dir, 'clustering', f'clusters_iter_{iteration}.json'), 'w') as f:
            json.dump(clustering_results, f, indent=4)
        
        # Calculate typicality scores
        typicality_scores = calculate_typicality(train_features)
        
        # Select samples using TPC strategy
        new_tpc_indices = select_examples(
            cluster_labels, 
            typicality_scores, 
            budget=args.budget, 
            labeled_indices=tpc_labeled_indices
        )
        
        # Add new samples to the labeled sets
        tpc_labeled_indices = np.append(tpc_labeled_indices, new_tpc_indices)
        
        # Select random samples (ensuring no overlap with existing random samples)
        available_indices = np.setdiff1d(np.arange(len(train_features)), random_labeled_indices)
        new_random_indices = np.random.choice(available_indices, args.budget, replace=False)
        random_labeled_indices = np.append(random_labeled_indices, new_random_indices)
        
        sampling_results = {
            'iteration': int(iteration),
            'tpc_new_indices': new_tpc_indices.tolist(),
            'random_new_indices': new_random_indices.tolist(),
            'tpc_all_indices': tpc_labeled_indices.tolist(),
            'random_all_indices': random_labeled_indices.tolist()
        }
        
        with open(os.path.join(rep_dir, 'results', f'sampling_iter_{iteration}.json'), 'w') as f:
            json.dump(sampling_results, f, indent=4)
        
        # Update current budget
        current_budget += args.budget
        
        # Create feature data loaders for both strategies
        tpc_train_loader, test_loader = get_feature_loaders(
            train_features, train_labels, 
            test_features, test_labels, 
            tpc_labeled_indices, args.batch_size
        )
        
        random_train_loader, _ = get_feature_loaders(
            train_features, train_labels, 
            test_features, test_labels, 
            random_labeled_indices, args.batch_size
        )
        
        # Train and evaluate TPC model
        print("Training TPC model...")
        tpc_model = LinearClassifier(feature_dim, NUM_CLASSES).to(device)
        criterion = nn.CrossEntropyLoss()
        # Learning rate increased by 100x compared to fully supervised (2.5 instead of 0.025)
        optimizer = optim.SGD(tpc_model.parameters(), lr=2.5, momentum=0.9, nesterov=True, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        
        tpc_model = train_model(tpc_model, tpc_train_loader, criterion, optimizer, scheduler, device, epochs=args.epochs)
        tpc_accuracy = evaluate_model(tpc_model, test_loader, device)
        print(f"TPC Model Accuracy: {tpc_accuracy:.4f}")
        
        # Train and evaluate Random model
        print("Training Random model...")
        random_model = LinearClassifier(feature_dim, NUM_CLASSES).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(random_model.parameters(), lr=2.5, momentum=0.9, nesterov=True, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        
        random_model = train_model(random_model, random_train_loader, criterion, optimizer, scheduler, device, epochs=args.epochs)
        random_accuracy = evaluate_model(random_model, test_loader, device)
        print(f"Random Model Accuracy: {random_accuracy:.4f}")
        
        # Update the results
        budgets.append(current_budget)
        tpc_accuracies.append(tpc_accuracy)
        random_accuracies.append(random_accuracy)
        
        # Save performance results for this iteration (only current iteration data)
        performance_results = {
            'iteration': int(iteration),
            'budget': int(args.budget),
            'cumulative_budget': int(current_budget),
            'tpc_accuracy': float(tpc_accuracy),
            'random_accuracy': float(random_accuracy),
            'accuracy_diff': float(tpc_accuracy - random_accuracy),
            'tpc_new_indices': new_tpc_indices.tolist() if isinstance(new_tpc_indices, np.ndarray) else new_tpc_indices,
            'random_new_indices': new_random_indices.tolist() if isinstance(new_random_indices, np.ndarray) else new_random_indices
        }
        
        with open(os.path.join(rep_dir, 'results', f'performance_iter_{iteration}.json'), 'w') as f:
            json.dump(performance_results, f, indent=4)
        
        # Update the plot
        update_plot(rep_dir, budgets, tpc_accuracies, random_accuracies, iteration)
        
        cumulative_results = {
            'budgets': [int(b) for b in budgets],
            'tpc_accuracies': [float(a) for a in tpc_accuracies],
            'random_accuracies': [float(a) for a in random_accuracies],
            'tpc_labeled_indices': tpc_labeled_indices.tolist() if isinstance(tpc_labeled_indices, np.ndarray) else tpc_labeled_indices,
            'random_labeled_indices': random_labeled_indices.tolist() if isinstance(random_labeled_indices, np.ndarray) else random_labeled_indices
        }
        
        with open(os.path.join(rep_dir, 'results', 'cumulative_results.json'), 'w') as f:
            json.dump(cumulative_results, f, indent=4)
        
        print(f"Results saved to {rep_dir}/results/")
    
    # Return final results for this repetition
    return {
        'budgets': budgets,
        'tpc_accuracies': tpc_accuracies,
        'random_accuracies': random_accuracies
    }

def main():
    # Parse arguments
    args = parse_args()
    
    # Create parent experiment directory
    parent_exp_dir = args.exp_dir
    os.makedirs(parent_exp_dir, exist_ok=True)
    
    # Save overall configuration
    with open(os.path.join(parent_exp_dir, 'config.json'), 'w') as f:
        config = vars(args)
        config['device'] = str(device)
        json.dump(config, f, indent=4)
    
    # Run experiments for each repetition
    all_results = []
    
    for rep in range(args.repetitions):
        print(f"\n\n{'='*80}")
        print(f"Starting Repetition {rep+1}/{args.repetitions}")
        print(f"{'='*80}\n")
        
        # Create directory for this repetition
        rep_dir = os.path.join(parent_exp_dir, f'rep_{rep}')
        
        # Set seed for this repetition
        rep_seed = args.seed + rep
        
        # Run the experiment
        results = run_single_experiment(args, rep_dir, rep_seed)
        all_results.append(results)
    
    # If we have multiple repetitions, create an aggregated plot
    if args.repetitions > 1:
        aggregated_results = create_aggregated_plot(parent_exp_dir, args.repetitions, args.iterations, args.budget)
    
    print("\nAll experiments completed successfully!")

if __name__ == "__main__":
    main() 
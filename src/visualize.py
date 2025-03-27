import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Define CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
              'dog', 'frog', 'horse', 'ship', 'truck']

def visualize_features(features, labels, perplexity=30, n_iter=1000, random_state=42, 
                      filename="feature_visualization"):
    print("Running t-SNE for dimensionality reduction...")
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=random_state)
    features_tsne = tsne.fit_transform(features)

    
    # Plot features colored by class
    plt.figure(figsize=(12, 10))
    
    # Create scatter plot
    scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], 
                         c=labels, cmap='tab10', alpha=0.7, s=10)
    
    # Add legend
    legend1 = plt.legend(*scatter.legend_elements(),
                        loc="upper right", title="Classes")
    plt.gca().add_artist(legend1)
    
    # Add annotations for class names
    for i, name in enumerate(class_names):
        # Find a point for this class to place the annotation
        idx = np.where(labels == i)[0][0]
        plt.annotate(name, 
                   (features_tsne[idx, 0], features_tsne[idx, 1]),
                   xytext=(5, 5),
                   textcoords='offset points',
                   fontsize=9,
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    
    plt.title("t-SNE Visualization of Feature Space")
    plt.tight_layout()
    
    # Create a directory for plots if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    # Save the plot
    plt.savefig(f"plots/{filename}.png", dpi=300, bbox_inches='tight')
    print(f"Plot saved to plots/{filename}.png")
    
    return features_tsne

def visualize_clusters(features, cluster_labels, true_labels=None, 
                      title="t-SNE Visualization of Clusters", filename="clusters_tsne"):
    # Use PCA to reduce dimensionality before t-SNE for faster computation
    print("Running PCA for dimensionality reduction...")
    pca = PCA(n_components=50)
    features_pca = pca.fit_transform(features)
    
    # Apply t-SNE to reduce to 2 dimensions for visualization
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    features_tsne = tsne.fit_transform(features_pca)
    
    # Create a directory for plots if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Plot clusters
    plt.figure(figsize=(12, 10))
    
    n_clusters = len(np.unique(cluster_labels))
    scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], 
               c=cluster_labels, cmap='tab10', alpha=0.7, s=10)
    
    plt.colorbar(scatter, label='Cluster')
    plt.title(f"{title} (K={n_clusters})")
    plt.tight_layout()
    
    # Save the clusters plot
    plt.savefig(f"plots/{filename}_clusters.png", dpi=300, bbox_inches='tight')
    print(f"Plot saved to plots/{filename}_clusters.png")
    
    # If true labels are provided, also create a plot with true classes
    if true_labels is not None:
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], 
                   c=true_labels, cmap='tab10', alpha=0.7, s=10)
        
        # Create custom legend
        handles, _ = scatter.legend_elements()
        plt.legend(handles, class_names, loc="upper right", title="Classes")
        
        plt.title(f"{title} (True Classes)")
        plt.tight_layout()
        
        # Save the true classes plot
        plt.savefig(f"plots/{filename}_true_classes.png", dpi=300, bbox_inches='tight')
        print(f"Plot saved to plots/{filename}_true_classes.png")
        
    # Create a visualization showing both cluster assignments and true classes if true labels are provided
    if true_labels is not None:
        # Create a 1x2 subplot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Plot clusters
        scatter1 = ax1.scatter(features_tsne[:, 0], features_tsne[:, 1], 
                   c=cluster_labels, cmap='tab10', alpha=0.7, s=10)
        ax1.set_title(f"Cluster Assignments (K={n_clusters})")
        fig.colorbar(scatter1, ax=ax1, label='Cluster')
        
        # Plot true classes
        scatter2 = ax2.scatter(features_tsne[:, 0], features_tsne[:, 1], 
                   c=true_labels, cmap='tab10', alpha=0.7, s=10)
        ax2.set_title("True Classes")
        
        # Create custom legend for classes
        handles, _ = scatter2.legend_elements()
        ax2.legend(handles, class_names, loc="upper right", title="Classes")
        
        plt.tight_layout()
        
        # Save the side-by-side comparison plot
        plt.savefig(f"plots/{filename}_comparison.png", dpi=300, bbox_inches='tight')
        print(f"Plot saved to plots/{filename}_comparison.png")
       
    return features_tsne

def visualize_selected_examples(features, cluster_labels, true_labels, selected_indices, 
                               typicality_scores, 
                               title="Selected Examples", filename="selected_examples"):
    print("Creating visualization of selected examples...")
    
    # Use PCA to reduce dimensionality before t-SNE for faster computation
    print("Running PCA for dimensionality reduction...")
    pca = PCA(n_components=50)
    features_pca = pca.fit_transform(features)
    
    # Apply t-SNE to reduce to 2 dimensions for visualization
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    features_tsne = tsne.fit_transform(features_pca)
    
    # Create a directory for plots if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Create a scatter plot showing clusters with selected examples highlighted
    plt.figure(figsize=(12, 10))
    
    # Plot all points colored by cluster
    scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], 
                c=cluster_labels, cmap='tab10', alpha=0.3, s=10)
    
    # Highlight selected examples with larger, red markers
    plt.scatter(features_tsne[selected_indices, 0], features_tsne[selected_indices, 1], 
               color='red', s=100, marker='*', edgecolors='black', linewidths=1)
    
    plt.colorbar(scatter, label='Cluster')
    plt.title(f"{title}")
    plt.tight_layout()
    
    # Save the clusters with selected examples plot
    plt.savefig(f"plots/{filename}_clusters.png", dpi=300, bbox_inches='tight')
    print(f"Plot saved to plots/{filename}_clusters.png")
    
    # Create a separate visualization showing selected examples with true class labels
    plt.figure(figsize=(12, 10))
    
    # Plot all points colored by true class
    scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], 
                c=true_labels, cmap='tab10', alpha=0.3, s=10)
    
    # Highlight selected examples with larger, red markers
    selected_points = plt.scatter(features_tsne[selected_indices, 0], features_tsne[selected_indices, 1], 
                    color='red', s=100, marker='*', edgecolors='black', linewidths=1)
    
    # Add legend for classes
    handles, _ = scatter.legend_elements()
    class_legend = plt.legend(handles, class_names, loc="upper right", title="Classes")
    plt.gca().add_artist(class_legend)
    
    # Add legend for selected examples
    plt.legend([selected_points], ["Selected Examples"], loc="lower right")
    
    plt.title(f"{title} (True Classes)")
    plt.tight_layout()
    
    # Save the true classes with selected examples plot
    plt.savefig(f"plots/{filename}_true_classes.png", dpi=300, bbox_inches='tight')
    print(f"Plot saved to plots/{filename}_true_classes.png")
   
    # Print information about selected examples
    print("\nSelected Examples Information:")
    print("=" * 80)
    print(f"{'Index':<10}{'Cluster':<10}{'True Class':<15}{'Typicality':<15}")
    print("-" * 80)
    
    for i, idx in enumerate(selected_indices):
        cluster = cluster_labels[idx]
        true_class = true_labels[idx]
        typicality = typicality_scores[idx]
        print(f"{idx:<10}{cluster:<10}{class_names[true_class]:<15}{typicality:.4f}")
    
    print("=" * 80)
    
    return features_tsne

def plot_learning_curves(iterations, budget_points, typiclust_accuracies, random_accuracies, 
                        title="Active Learning Performance", filename="learning_curves"):
    
    plt.figure(figsize=(10, 6))
    
    # Plot both learning curves
    plt.plot(budget_points, typiclust_accuracies, 'b-o', label='TypiClust')
    plt.plot(budget_points, random_accuracies, 'r-s', label='Random')
    
    # Add labels and title
    plt.xlabel('Number of Labeled Examples (Budget)')
    plt.ylabel('Test Accuracy')
    plt.title(title)
    
    # Add grid and legend
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Ensure all budget points are shown on x-axis
    plt.xticks(budget_points)
    
    # Set y-axis to start from 0
    plt.ylim(bottom=0)
    
    plt.tight_layout()
    
    # Create a directory for plots if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Save the learning curves plot
    plt.savefig(f"plots/{filename}.png", dpi=300, bbox_inches='tight')
    print(f"Plot saved to plots/{filename}.png")
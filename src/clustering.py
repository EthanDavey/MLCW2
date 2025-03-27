import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
from tqdm import tqdm

# CIFAR-10 class names for reference
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
              'dog', 'frog', 'horse', 'ship', 'truck']

def load_features(features_path):
    """Load features and labels from a pickle file."""
    print(f"Loading features from {features_path}...")
    with open(features_path, 'rb') as f:
        data = pickle.load(f)
    return data['features'], data['labels']

def perform_kmeans(features, n_clusters, random_state=42):
    """Perform k-means clustering on the features."""
    print(f"Performing k-means with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10, verbose=0)
    cluster_labels = kmeans.fit_predict(features)
    
    return cluster_labels, kmeans.cluster_centers_, kmeans

def visualize_clusters_tsne(features, cluster_labels, true_labels=None, title="t-SNE Visualization of Clusters", filename="clusters_tsne"):
    """Visualize clusters using t-SNE."""
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
    
    # Save the plot
    plt.savefig(f'plots/{filename}_clusters.png', dpi=300, bbox_inches='tight')
    print(f"Cluster visualization saved as plots/{filename}_clusters.png")
    
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
        
        # Save the plot
        plt.savefig(f'plots/{filename}_true_classes.png', dpi=300, bbox_inches='tight')
        print(f"True classes visualization saved as plots/{filename}_true_classes.png")
    
    # Create a visualization showing both cluster assignments and true classes
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
        
        # Save the plot
        plt.savefig(f'plots/{filename}_comparison.png', dpi=300, bbox_inches='tight')
        print(f"Comparison visualization saved as plots/{filename}_comparison.png")

def analyze_cluster_composition(cluster_labels, true_labels):
    """Analyze the composition of each cluster in terms of true classes."""
    n_clusters = len(np.unique(cluster_labels))
    n_classes = len(np.unique(true_labels))
    
    # Create a matrix to store the counts
    composition = np.zeros((n_clusters, n_classes), dtype=int)
    
    # Count the occurrences of each class in each cluster
    for i in range(len(cluster_labels)):
        cluster = cluster_labels[i]
        true_class = true_labels[i]
        composition[cluster, true_class] += 1
    
    # Calculate percentage composition
    percentage = composition / composition.sum(axis=1, keepdims=True) * 100
    
    # Print the composition of each cluster
    print("\nCluster Composition (percentage of each class in the cluster):")
    print("=" * 80)
    print(f"{'Cluster':<10}", end="")
    for class_idx, class_name in enumerate(class_names):
        print(f"{class_name:<10}", end="")
    print()
    print("-" * 80)
    
    for cluster in range(n_clusters):
        print(f"{cluster:<10}", end="")
        for class_idx in range(n_classes):
            print(f"{percentage[cluster, class_idx]:.1f}%    ", end="")
        
        # Find the dominant class
        dominant_class = np.argmax(composition[cluster])
        dominant_pct = percentage[cluster, dominant_class]
        
        print(f" | Dominant: {class_names[dominant_class]} ({dominant_pct:.1f}%)")
    
    print("=" * 80)
    
    # Return the composition and percentage for further analysis
    return composition, percentage

def save_clustering_results(features, cluster_labels, centroids, kmeans_model, labeled_indices=None):
    
    results = {
        'cluster_labels': cluster_labels,
        'centroids': centroids,
        'labeled_indices': labeled_indices,
        'kmeans_model': kmeans_model
    }
    

def main():
    # Configuration
    B = 10  # Budget for active learning
    features_path = 'features/train_features.pkl'
    
    # Load features and labels
    features, true_labels = load_features(features_path)
    
    # For initial iteration, |L_0| = 0, so n_clusters = B
    n_clusters = B
    
    # Perform k-means clustering
    cluster_labels, centroids, kmeans_model = perform_kmeans(features, n_clusters)
    
    # Visualize the clusters using t-SNE
    # Use a subset of data for visualization if the dataset is large
    max_vis_samples = 10000
    if len(features) > max_vis_samples:
        print(f"Using {max_vis_samples} random samples for visualization...")
        indices = np.random.choice(len(features), max_vis_samples, replace=False)
        vis_features = features[indices]
        vis_cluster_labels = cluster_labels[indices]
        vis_true_labels = true_labels[indices]
    else:
        vis_features = features
        vis_cluster_labels = cluster_labels
        vis_true_labels = true_labels
    
    '''visualize_clusters_tsne(vis_features, vis_cluster_labels, vis_true_labels, 
                           title="CIFAR-10 Clusters from SimCLR Features", 
                           filename="cifar10_kmeans")'''
    
    # Analyze the composition of each cluster
    analyze_cluster_composition(cluster_labels, true_labels)
    
    # Save the clustering results
    save_clustering_results(features, cluster_labels, centroids, kmeans_model)

if __name__ == "__main__":
    main() 
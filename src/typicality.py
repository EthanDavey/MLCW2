import numpy as np
from sklearn.neighbors import NearestNeighbors


def calculate_typicality(features, k=20):
    print(f"Calculating typicality scores using {k} nearest neighbors...")
    
    # Fit nearest neighbors model
    nn = NearestNeighbors(n_neighbors=k+1, algorithm='auto', metric='euclidean')  # k+1 because the point itself is included
    nn.fit(features)
    
    # Calculate distances to k nearest neighbors for each point
    distances, _ = nn.kneighbors(features)
    
    # Skip the first distance (distance to self = 0)
    # and calculate average distance to k nearest neighbors
    avg_distances = np.mean(distances[:, 1:], axis=1)
    
    # Typicality is inverse of average distance
    typicality_scores = 1.0 / avg_distances
    
    return typicality_scores

def select_examples(cluster_labels, typicality_scores, budget=10, labeled_indices=None):
    n_clusters = len(np.unique(cluster_labels))
    
    # Determine which examples are unlabeled
    if labeled_indices is None or len(labeled_indices) == 0:
        labeled_indices = np.array([], dtype=int)  # Empty array for initial selection
    
    # Count number of points in each cluster
    cluster_sizes = np.zeros(n_clusters, dtype=int)
    for i in range(n_clusters):
        cluster_sizes[i] = np.sum(cluster_labels == i)
    
    # Identify clusters that already have labeled examples
    covered_clusters = set()
    for idx in labeled_indices:
        covered_clusters.add(cluster_labels[idx])
    
    # Identify clusters that don't have any labeled examples yet
    uncovered_clusters = set(range(n_clusters)) - covered_clusters
    print(f"Number of uncovered clusters: {len(uncovered_clusters)}")
    
    # Sort uncovered clusters by size (largest first)
    uncovered_clusters_list = list(uncovered_clusters)
    uncovered_clusters_list.sort(key=lambda c: cluster_sizes[c], reverse=True)
    
    # Select the B largest uncovered clusters
    selected_clusters = uncovered_clusters_list[:budget]
    print(f"Selected clusters: {selected_clusters}")
    
    # For each selected cluster, find the most typical unlabeled example
    selected_indices = []
    for cluster in selected_clusters:
        # Get the points in this cluster (they should all be unlabeled)
        cluster_indices = np.where(cluster_labels == cluster)[0]
        
        if len(cluster_indices) > 0:
            # Find the most typical example in this cluster
            cluster_typicality = typicality_scores[cluster_indices]
            most_typical_idx = cluster_indices[np.argmax(cluster_typicality)]
            selected_indices.append(most_typical_idx)
        else:
            print(f"Warning: No unlabeled examples in cluster {cluster}")
    
    print(f"Selected {len(selected_indices)} examples")
    return np.array(selected_indices)
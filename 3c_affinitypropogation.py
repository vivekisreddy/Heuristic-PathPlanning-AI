import numpy as np
from sklearn.datasets import load_digits
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import confusion_matrix, fowlkes_mallows_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def initialize_centroids(data, k):
    """
    Randomly initialize centroids.

    Parameters:
    - data (numpy.ndarray): Input data with shape (n_samples, n_features).
    - k (int): Number of clusters.

    Returns:
    - numpy.ndarray: Initial centroids with shape (k, n_features).
    """
    return data[np.random.choice(data.shape[0], k, replace=False)]

def nearest_points(data, centroids):
    """
    Assign each point to the nearest centroid using Euclidean distance.

    Parameters:
    - data (numpy.ndarray): Input data with shape (n_samples, n_features).
    - centroids (numpy.ndarray): Centroids with shape (k, n_features).

    Returns:
    - tuple: Tuple containing a list of clusters and an array of cluster labels.
    """
    distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
    labels = np.argmin(distances, axis=1)
    clusters = [data[labels == i] for i in range(len(centroids))]
    return clusters, labels

def new_centroids(clusters):
    """
    Update centroids based on the mean of points in each cluster.

    Parameters:
    - clusters (list): List of clusters, where each cluster is a numpy.ndarray.

    Returns:
    - numpy.ndarray: Updated centroids with shape (k, n_features).
    """
    return np.array([np.mean(cluster, axis=0) for cluster in clusters])

def agglomerative_clustering(data, k):
    """
    Perform agglomerative clustering using the Ward linkage.

    Parameters:
    - data (numpy.ndarray): Input data with shape (n_samples, n_features).
    - k (int): Number of clusters.

    Returns:
    - numpy.ndarray: Cluster labels assigned by the Agglomerative Clustering algorithm.
    """
    model = AgglomerativeClustering(n_clusters=k, linkage='ward')
    labels = model.fit_predict(data)
    return labels

def affinity_propagation_clustering(data):
    """
    Perform affinity propagation clustering.

    Parameters:
    - data (numpy.ndarray): Input data with shape (n_samples, n_features).

    Returns:
    - numpy.ndarray: Cluster labels assigned by the Affinity Propagation algorithm.
    """
    model = AffinityPropagation()
    labels = model.fit_predict(data)
    return labels

def get_majority_label(cluster_indices, true_labels):
    """
    Get the majority label for a cluster.

    Parameters:
    - cluster_indices (numpy.ndarray): Indices of data points in a cluster.
    - true_labels (numpy.ndarray): True labels for all data points.

    Returns:
    - int: Majority label for the cluster.
    """
    cluster_labels = true_labels[cluster_indices]
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    majority_label = unique_labels[np.argmax(counts)]
    return majority_label


def evaluate_affinity_propagation(labels, labels_true, data):
    """
    Evaluate affinity propagation clustering performance and visualize the results.

    Parameters:
    - labels (numpy.ndarray): Cluster labels assigned by the clustering algorithm.
    - labels_true (numpy.ndarray): True labels for all data points.
    - data (numpy.ndarray): Input data with shape (n_samples, n_features).
    """
    # Create a mapping from cluster index to majority label
    cluster_to_majority_label = {}
    for cluster in np.unique(labels):
        cluster_indices = np.where(labels == cluster)[0]
        majority_label = get_majority_label(cluster_indices, labels_true)
        cluster_to_majority_label[cluster] = majority_label

    # Use the mapping to generate predicted labels for each sample
    predicted_labels_corrected = np.array([cluster_to_majority_label[cluster] for cluster in labels])

    # Report the 10x10 confusion matrix
    confusion_mat = confusion_matrix(labels_true, predicted_labels_corrected, labels=np.arange(10))
    print("Confusion Matrix:")
    print(confusion_mat)

    # Calculate Fowlkes-Mallows index
    fowlkes_mallows_index = fowlkes_mallows_score(labels_true, predicted_labels_corrected)
    print(f"Fowlkes-Mallows Index: {fowlkes_mallows_index}")

    # Visualize the clustering results (scatter plot)
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data)
    
    plt.figure(figsize=(8, 6))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(np.unique(labels))))
    for cluster, color in zip(np.unique(labels), colors):
        cluster_indices = np.where(labels == cluster)[0]
        plt.scatter(data_pca[cluster_indices, 0], data_pca[cluster_indices, 1], color=color, label=f"Cluster {cluster}")

    plt.title("Affinity Propagation Clustering Results")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.show()

    # Print corrected cluster assignments
    print("\nCorrected Cluster Assignments:")
    for cluster, majority_label in cluster_to_majority_label.items():
        print(f"Cluster {cluster} is assigned to Label {majority_label}")

def main():
    """
    Main function to execute clustering on the provided dataset and evaluate the results.
    """
    # Load the handwritten digits dataset
    digits = load_digits()
    data = digits.data
    labels_true = digits.target

    # Set the number of clusters (digits)
    k = 10

    # Perform Affinity Propagation Clustering
    labels_affinity_propagation = affinity_propagation_clustering(data)

    # Evaluate Affinity Propagation clustering performance and plot the results
    evaluate_affinity_propagation(labels_affinity_propagation, labels_true, data)

if __name__ == "__main__":
    main()

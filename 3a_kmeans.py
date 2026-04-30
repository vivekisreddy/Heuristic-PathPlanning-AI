import numpy as np
import matplotlib.pyplot as plt
from kmeans_algo import k_means

from sklearn.datasets import load_digits
from sklearn.metrics import fowlkes_mallows_score, confusion_matrix

def majority_voting(labels, true_labels, num_clusters):
    """
    Assign majority labels to each cluster based on the true labels.

    Parameters:
    - labels (numpy.ndarray): Cluster labels assigned by the clustering algorithm.
    - true_labels (numpy.ndarray): True labels for all data points.
    - num_clusters (int): Number of clusters.

    Returns:
    - tuple: Tuple containing assigned labels for each data point and mapping between clusters and majority labels.
    """
    cluster_labels = np.zeros(num_clusters, dtype=int)

    for cluster in range(num_clusters):
        cluster_indices = np.where(labels == cluster)[0]
        cluster_true_labels = true_labels[cluster_indices]
        unique_labels, counts = np.unique(cluster_true_labels, return_counts=True)
        majority_label = unique_labels[np.argmax(counts)]
        cluster_labels[cluster] = majority_label

    assigned_labels = cluster_labels[labels]
    return assigned_labels, cluster_labels

def assess_clustering(labels, true_labels):
    """
    Evaluate clustering performance using the Fowlkes-Mallows index and confusion matrix.

    Parameters:
    - labels (numpy.ndarray): Cluster labels assigned by the clustering algorithm.
    - true_labels (numpy.ndarray): True labels for all data points.

    Returns:
    - tuple: Tuple containing the Fowlkes-Mallows index and confusion matrix.
    """
    fmi = fowlkes_mallows_score(true_labels, labels)
    conf_matrix = confusion_matrix(true_labels, labels, labels=range(10))
    return fmi, conf_matrix

def main():
    """
    Main function to execute k-means clustering on the provided dataset, evaluate the results, and visualize centroids.
    """
    digits = load_digits()
    data = digits.data
    true_labels = digits.target

    k = 10

    # Perform k-means clustering
    labels, centroids = k_means(data, k)

    # Assign majority labels to clusters
    assigned_labels, cluster_labels = majority_voting(labels, true_labels, k)

    # Evaluate clustering performance
    fmi, conf_matrix = assess_clustering(assigned_labels, true_labels)

    # Print evaluation metrics
    print(f"Fowlkes-Mallows Index: {fmi}")
    print("Confusion Matrix:")
    print(conf_matrix)

    # Print the mapping between clusters and majority labels
    print("\nCluster to Majority Label Mapping:")
    for cluster, majority_label in enumerate(cluster_labels):
        print(f"Cluster {cluster+1} is assigned to Label {majority_label}")

    # Visualize centroids
    fig, ax = plt.subplots(2, 5, figsize=(8, 4))
    ax = ax.flatten()

    for i, centroid in enumerate(centroids):
        centroid_image = centroid.reshape(8, 8)
        ax[i].imshow(centroid_image, cmap='gray')
        ax[i].set_title(f'Cluster {i+1}')

    plt.show()

if __name__ == "__main__":
    main()

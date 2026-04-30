# kmeans_algo.py

import numpy as np
from math import sqrt

def euclidean_distance(point1, point2):
    """
    Calculate Euclidean distance between two points.

    Parameters:
    - point1 (tuple): Coordinates of the first point (x1, y1).
    - point2 (tuple): Coordinates of the second point (x2, y2).

    Returns:
    - float: Euclidean distance between the two points.
    """
    return sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

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

def assign_to_clusters(data, centroids):
    """
    Assign each data point to the nearest centroid.

    Parameters:
    - data (numpy.ndarray): Input data with shape (n_samples, n_features).
    - centroids (numpy.ndarray): Centroids with shape (k, n_features).

    Returns:
    - numpy.ndarray: Array of cluster labels for each data point.
    """
    distances = np.linalg.norm(data - centroids[:, np.newaxis], axis=2)
    return np.argmin(distances, axis=0)

def calculate_new_centroids(data, labels, k):
    """
    Update centroids based on the mean of points in each cluster.

    Parameters:
    - data (numpy.ndarray): Input data with shape (n_samples, n_features).
    - labels (numpy.ndarray): Array of cluster labels for each data point.
    - k (int): Number of clusters.

    Returns:
    - numpy.ndarray: Updated centroids with shape (k, n_features).
    """
    return np.array([data[labels == i].mean(axis=0) for i in range(k)])

def has_converged(centroids, new_centroids, tolerance=1e-4):
    """
    Check if the k-means algorithm has converged.

    Parameters:
    - centroids (numpy.ndarray): Current centroids.
    - new_centroids (numpy.ndarray): Updated centroids.
    - tolerance (float): Convergence threshold.

    Returns:
    - bool: True if the algorithm has converged, False otherwise.
    """
    return np.all(np.abs(centroids - new_centroids) < tolerance)

def k_means(data, k, max_iters=1000, tolerance=1e-4):
    """
    Perform k-means clustering.

    Parameters:
    - data (numpy.ndarray): Input data with shape (n_samples, n_features).
    - k (int): Number of clusters.
    - max_iters (int): Maximum number of iterations for the k-means algorithm.
    - tolerance (float): Convergence threshold.

    Returns:
    - tuple: Tuple containing cluster labels and final centroids.
    """
    # Step 1: Initialize centroids
    centroids = initialize_centroids(data, k)

    # Initialize new_centroids to avoid UnboundLocalError
    new_centroids = np.zeros_like(centroids)

    # Iterate through a maximum number of iterations
    for iteration in range(max_iters):
        # Step 2: Assign each data point to the nearest centroid
        labels = assign_to_clusters(data, centroids)

        # Step 3: Update centroids based on the mean of points in each cluster
        new_centroids = calculate_new_centroids(data, labels, k)

        # Step 4: Check for convergence
        if has_converged(centroids, new_centroids, tolerance):
            print(f"Converged after {iteration + 1} iterations.")
            break

        # Update centroids for the next iteration
        centroids = new_centroids

    return labels, centroids


import numpy as np
import matplotlib.pyplot as plt
from kmeans_algo import k_means

def main():
    """
    Main function to execute k-means clustering on the provided data and visualize the results.
    """
    # Load data from the file using numpy's loadtxt function
    data = np.loadtxt('cluster_data.txt', delimiter='\t')

    # Specify the number of clusters (you can adjust this to have how many ever clusters)
    k = 3

    # Run k-means algorithm on the loaded data
    labels, centroids = k_means(data, k)

    # Plot the data points with different colors and symbols for each cluster
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', marker='o')

    # Plot centroids with a different marker and color
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')

    # Label the axes of the plot
    plt.xlabel('Length')
    plt.ylabel('Width')

    # Add legend for clusters to distinguish which cluster is which 
    for i in range(k):
        plt.scatter([], [], label=f'Cluster {i+1}', color=plt.cm.viridis(i/k))

    # Display legend
    plt.legend()

    # Set the title of the plot
    plt.title('K-Means Clustering')

    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()

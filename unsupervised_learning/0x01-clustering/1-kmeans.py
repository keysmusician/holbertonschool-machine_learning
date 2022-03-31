#!/usr/bin/env python3
""" Defines `kmeans`. """
import numpy as np


def kmeans(X, cluster_count, iterations=1000):
    """
    Performs K-means on a dataset.

    X: A numpy.ndarray of shape (n, d) containing the dataset that will be used
        for K-means clustering.
        n: The number of data points.
        d: The number of dimensions for each data point.
    k: A positive integer containing the number of clusters.
    iterations: A positive integer containing the maximum number of iterations
        that should be performed.

    Returns: (C, clss) on success, or (None, None) on failure.
        C: A numpy.ndarray of shape (k, d) containing the centroid means for
            each cluster.
        clss: A numpy.ndarray of shape (n,) containing the index of the cluster
            in C that each data point belongs to.
    """
    if (
            type(X) is not np.ndarray or
            len(X.shape) != 2 or
            type(cluster_count) is not int or
            cluster_count < 1 or
            type(iterations) is not int or
            iterations < 1
            ):
        return (None, None)

    _, d = X.shape

    minimums = np.amin(X, axis=0)
    maximums = np.amax(X, axis=0)
    centroids = np.random.uniform(minimums, maximums, (cluster_count, d))

    for iteration in range(iterations):
        previous_centroids = np.copy(centroids)
        distances = np.linalg.norm(X - centroids[:, np.newaxis], axis=2)
        cluster_labels = np.argmin(distances, axis=0)
        cluster_sizes = np.bincount(cluster_labels, minlength=cluster_count)

        for cluster_index in range(cluster_count):
            if cluster_sizes[cluster_index] == 0:
                centroids[cluster_index] = np.random.uniform(
                    minimums, maximums, (1, d))
            else:
                centroids[cluster_index] = np.mean(
                    X[cluster_labels == cluster_index], axis=0)
        if np.array_equal(previous_centroids, centroids):
            break

    distances = np.linalg.norm(X - centroids[:, np.newaxis], axis=2)
    cluster_labels = np.argmin(distances, axis=0)

    return (centroids, cluster_labels)

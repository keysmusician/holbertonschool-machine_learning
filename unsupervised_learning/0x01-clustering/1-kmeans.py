#!/usr/bin/env python3
""" Defines `kmeans`. """
import numpy as np


def kmeans(X, k, iterations=1000):
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
            type(k) is not int or
            k < 1 or
            type(iterations) is not int or
            iterations < 1
            ):
        return (None, None)

    _, d = X.shape

    minimums = np.amin(X, axis=0)
    maximums = np.amax(X, axis=0)
    centroids = np.random.uniform(minimums, maximums, (k, d))

    for iteration in range(iterations):
        prev_centroids = np.copy(centroids)
        centered_points = centroids[:, np.newaxis] - X
        distances = np.linalg.norm(centered_points, axis=2)
        cluster_labels = np.argmin(distances, axis=0)
        cluster_sizes = np.bincount(cluster_labels, minlength=k)

        empty_clusters = np.where(cluster_sizes == 0)
        nonempty_clusters = np.nonzero(cluster_sizes)[0]
        centroids[empty_clusters] = np.random.uniform(
            minimums, maximums, (len(empty_clusters), d))

        mask = np.indices(centered_points.shape)[0] == \
            cluster_labels[:, np.newaxis]

        centroids[nonempty_clusters] = (
            np.sum(X * mask, axis=1)[nonempty_clusters] /
            cluster_sizes[:, np.newaxis][nonempty_clusters]
        )

        if np.array_equal(centroids, prev_centroids):
            break

    return (centroids, cluster_labels)

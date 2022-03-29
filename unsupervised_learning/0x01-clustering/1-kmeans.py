#!/usr/bin/env python3
""" Defines `initialize`. """
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
    if not type(k) is int or \
            k <= 0 or \
            not type(X) is np.ndarray or \
            len(X.shape) != 2:
        return

    n, d = X.shape

    minimums = np.amin(X, axis=0)
    maximums = np.amax(X, axis=0)
    centroids = np.random.uniform(minimums, maximums, (k, d))

    cluster_labels = np.zeros(n)
    for iteration in range(iterations):
        new_centroids = np.zeros_like(centroids)
        centroid_indexes = list(range(k))
        category_count = np.zeros((k, 1))

        for i, datum in enumerate(X):
            distances = np.linalg.norm(centroids - datum, axis=1)
            label = np.argmin(distances)
            cluster_labels[i] = label
            new_centroids[label] += datum
            category_count[label] += 1
            try:
                centroid_indexes.remove(label)
            except ValueError:
                pass
        new_centroids /= category_count
        new_centroids[centroid_indexes] = np.random.uniform(
            minimums, maximums, (len(centroid_indexes), d))

        if np.all(new_centroids == centroids):
            break
        else:
            centroids = new_centroids

    return centroids, cluster_labels

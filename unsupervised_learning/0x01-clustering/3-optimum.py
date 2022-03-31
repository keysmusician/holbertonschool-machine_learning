#!/usr/bin/env python3
""" Defines `optimum_k`. """
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Tests for the optimum number of clusters by variance.

    X: A numpy.ndarray of shape (n, d) containing the data set.
    kmin: A positive integer containing the minimum number of clusters to check
        for (inclusive).
    kmax: A positive integer containing the maximum number of clusters to check
        for (inclusive).
    iterations: A positive integer containing the maximum number of iterations
        for K-means.


    Returns: (results, d_vars) on success, or (None, None) on failure.
        results: A list containing the outputs of K-means for each cluster size
        d_vars: A list containing the difference in variance from the smallest
            cluster size for each cluster size.
    """
    if (
            type(X) is not np.ndarray or
            len(X.shape) != 2 or
            type(kmin) is not int or
            kmin < 1 or
            type(kmax) is not int or
            kmax < 1 or
            type(iterations) is not int or
            iterations < 1
            ):
        return (None, None)

    results = []
    variance_difference = []

    if kmax is None:
        kmax = X.shape[0]
    for k in range(kmin, kmax + 1):
        centroids_labels = kmeans(X, k, iterations)
        results.append(centroids_labels)
        variance_difference.append(variance(X, centroids_labels[0]))

    return (results, variance_difference[0] - variance_difference)

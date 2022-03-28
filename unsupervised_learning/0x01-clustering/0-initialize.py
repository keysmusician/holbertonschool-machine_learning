#!/usr/bin/env python3
""" Defines `initialize`. """
import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means.

    X: A numpy.ndarray of shape (n, d) containing the dataset that will be used
        for K-means clustering.
        n: The number of data points.
        d: The number of dimensions for each data point.
    k: A positive integer containing the number of clusters.

    Returns: A numpy.ndarray of shape (k, d) containing the initialized
        centroids for each cluster, or None on failure.
    """
    n, d = X.shape
    min = np.amin(X, axis=0)
    max = np.amax(X, axis=0)
    return np.random.uniform(min, max, (k, d))

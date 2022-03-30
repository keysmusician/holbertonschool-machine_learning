#!/usr/bin/env python3
""" Defines `initialize`. """
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    Initializes variables for a Gaussian Mixture Model.

    X: A numpy.ndarray of shape (n, d) containing the data set.
    k: A positive integer containing the number of clusters.

    Returns: (priors, centroids, covariances) on success, or (None, None, None)
        on failure.
        priors: A numpy.ndarray of shape (k,) containing the priors for each
            cluster, initialized evenly.
        centroids: A numpy.ndarray of shape (k, d) containing the centroid
            means for each cluster, initialized with K-means.
        covariances: A numpy.ndarray of shape (k, d, d) containing the
            covariance matrices for each cluster, initialized as identity
            matrices.
    """
    if (type(X) is not np.ndarray or
            len(X.shape) != 2 or
            type(k) is not int or
            k < 1):
        return (None, None, None)
    priors = np.repeat(1/k, k)
    centroids, class_labels = kmeans(X, k)
    covariances = np.repeat(
        np.identity(X.shape[1])[np.newaxis], k, axis=0)

    return (priors, centroids, covariances)

#!/usr/bin/env python3
""" Defines `kmeans`. """
import sklearn.cluster


def kmeans(X, k):
    """
    Performs K-means on a dataset.

    X: A numpy.ndarray of shape (n, d) containing the dataset.
    k: The number of clusters.

    Returns: (C, clss):
        C: A numpy.ndarray of shape (k, d) containing the centroid means for
            each cluster.
        clss: A numpy.ndarray of shape (n,) containing the index of the cluster
            in C that each data point belongs to.
    """
    return sklearn.cluster.k_means(X, k)[:2]

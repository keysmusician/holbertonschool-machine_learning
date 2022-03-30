#!/usr/bin/env python3
""" Defines `variance`. """
import numpy as np


def variance(X, C):
    """
    That calculates the total intra-cluster variance for a data set.

    X: A numpy.ndarray of shape (n, d) containing the data set.
    C: A numpy.ndarray of shape (k, d) containing the centroid means for each
        cluster.

    Returns: The total variance, or None on failure.
    """
    if (
            type(X) is not np.ndarray or
            type(C) is not np.ndarray or
            len(X.shape) != 2 or
            len(C.shape) != 2 or
            X.shape[1] != C.shape[1]
            ):
        return
    distances = np.linalg.norm(C[:, np.newaxis] - X, axis=2)
    cluster_labels = np.argmin(distances, axis=0)
    mask = np.indices(distances.shape)[0] == cluster_labels
    squared_distances = np.square(distances * mask)
    return np.sum(np.sum(squared_distances, axis=1))

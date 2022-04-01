#!/usr/bin/env python3
""" Defines `maximization`. """
import numpy as np


def maximization(X, g):
    """
    Calculates the maximization step in the EM algorithm for a GMM.

    X: A numpy.ndarray of shape (n, d) containing the data set.
    g: A numpy.ndarray of shape (k, n) containing the posterior
        probabilities for each data point in each cluster.

    Returns: (pi, m, S) on success, or (None, None, None) on failure:
        pi: A numpy.ndarray of shape (k,) containing the updated priors for
            each cluster.
        m: A numpy.ndarray of shape (k, d) containing the updated centroid
            means for each cluster.
        S: A numpy.ndarray of shape (k, d, d) containing the updated covariance
            matrices for each cluster.
    """
    if (
            type(X) is not np.ndarray or
            len(X.shape) != 2 or
            type(g) is not np.ndarray or
            len(g.shape) != 2
            ):
        return (None, None, None)
    n, d = X.shape
    k, _gn = g.shape
    if (
            _gn != n or
            not np.allclose(np.sum(g, axis=0), np.ones(n))
            ):
        return (None, None, None)
    g_sums = np.sum(g, axis=1, keepdims=True)
    centroids = g @ X / g_sums
    priors = g_sums / n
    centered_X = (X - centroids[:, np.newaxis])
    covariances = np.ones((k, d, d))
    covariances = (
        g[:, np.newaxis] *
        np.transpose(centered_X, axes=(0, 2, 1)) @
        centered_X / g_sums[:, np.newaxis]
    )

    return (priors.flatten(), centroids, covariances)

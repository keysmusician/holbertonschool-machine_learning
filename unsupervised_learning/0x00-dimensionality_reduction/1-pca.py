#!/usr/bin/env python3
"""Defines `pca`."""
import numpy as np


def pca(X, ndim):
    """
    Performs PCA on a dataset.

    X: Q numpy.ndarray of shape (n, d):
        n: The number of data points.
        d: The number of dimensions in each point.
    ndim: The new dimensionality of the transformed X.

    Returns: T, a numpy.ndarray of shape (n, ndim) containing the transformed
        version of X.
    """
    X_m = X - np.mean(X, axis=0)
    U, Σ, VT = np.linalg.svd(X_m)

    return X_m @ VT[:ndim].T

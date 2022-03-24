#!/usr/bin/env python3
"""Defines `pca`."""
import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on a dataset.

    X: A numpy.ndarray of shape (n, d):
        n: The number of data points.
        d: The number of dimensions in each point. All dimensions have a mean
            of 0 across all data points.
    var: The fraction of the variance that the PCA transformation should
        maintain.

    Returns: The weights matrix, W, that maintains var fraction of X's original
        variance.

        W: A numpy.ndarray of shape (d, nd) where nd is the new dimensionality
            of the transformed X.
    """
    U, Σ, VT = np.linalg.svd(X)
    Σ_percent = np.cumsum(Σ) / np.sum(Σ)
    reduced_dimentionality = 0
    while Σ_percent[reduced_dimentionality] < var:
        reduced_dimentionality += 1

    return VT.T[:, :reduced_dimentionality + 1]

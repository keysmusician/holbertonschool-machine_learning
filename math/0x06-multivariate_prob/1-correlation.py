#!/usr/bin/env python3
"""Defines `mean_cov`."""
import numpy as np


def correlation(C):
    """
    Calculates a correlation matrix.

    C: A numpy.ndarray of shape (d, d) containing a covariance matrix:
        d: The number of dimensions

    Returns: A numpy.ndarray of shape (d, d) containing the correlation matrix.
    """
    if not type(C) is np.ndarray:
        raise TypeError('C must be a numpy.ndarray')
    elif len(C.shape) != 2:
        raise ValueError('C must be a 2D square matrix')
    d0, d1 = C.shape
    if d0 != d1:
        raise ValueError("C must be a 2D square matrix")

    root_variances = np.sqrt(np.diag(C))

    return C / np.outer(root_variances, root_variances)

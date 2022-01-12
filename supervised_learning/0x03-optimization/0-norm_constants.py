#!/usr/bin/env python3
"""Defines `normalization_constants`."""
import numpy as np


def normalization_constants(X):
    """
    Calculates the normalization (standardization) constants of a matrix.

    Args:
        X: A numpy.ndarray of shape (m, nx) to normalize, where 'm' is the
            number of data points 'nx' is the number of features.

    Returns: The mean and standard deviation of each feature, respectively.
    """
    return (np.mean(X, axis=0), np.std(X, axis=0))

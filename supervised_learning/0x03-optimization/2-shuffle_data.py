#!/usr/bin/env python3
"""Defines `shuffle_data`."""
import numpy as np


def shuffle_data(X, Y):
    """
    Shuffles the data points in two matrices the same way.

    Args:
        X: A numpy.ndarray of shape (m, nx) to shuffle, where 'm' is the number
            of data points and 'nx' is the number of features in X.
        Y: A numpy.ndarray of shape (m, ny) to shuffle, where 'm' is the same
            number of data points as in X and 'ny' is the number of features in
            Y.

    Returns: The shuffled X and Y matrices.
    """
    if (X.shape[0] == Y.shape[0]):
        random_order = np.random.permutation(X.shape[0])
        return X[random_order], Y[random_order]

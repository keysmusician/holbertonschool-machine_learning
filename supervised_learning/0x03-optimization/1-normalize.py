#!/usr/bin/env python3
"""Defines `normalize`."""


def normalize(X, m, s):
    """
    Normalizes (standardizes) a matrix.

    Args:

        X: A numpy.ndarray of shape (d, nx) to normalize, where 'd' is the
            number of data points, and 'nx' is the number of features.
        m: A numpy.ndarray of shape (nx,) that contains the mean of all
            features of X
        s: A numpy.ndarray of shape (nx,) that contains the standard deviation
            of all features of X

    Returns: The normalized X matrix.
    """
    return (X - m) / s

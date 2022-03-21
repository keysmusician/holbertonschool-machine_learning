#!/usr/bin/env python3
"""Defines `mean_cov`."""
import numpy as np


def mean_cov(X):
    """
    Calculates the mean and covariance of a data set.

    X: A numpy.ndarray of shape (n, d) containing the data set:
        n: The number of data points
        d: The number of dimensions in each data point

    Returns: A 2-tuple of (mean, cov):
        mean: numpy.ndarray of shape (1, d) containing the mean of the data
            set
        cov: A numpy.ndarray of shape (d, d) containing the covariance matrix
            of the data set
    """
    if not type(X) is np.ndarray or X.ndim != 2:
        raise TypeError('X must be a 2D numpy.ndarray')
    elif X.shape[0] < 2:
        raise ValueError('X must contain multiple data points')

    variable_count = X.shape[1]
    covariance = np.zeros((variable_count, variable_count))
    means = np.mean(X, axis=0, keepdims=True)
    centered_variables = X - means
    for i in range(variable_count):
        for j in range(variable_count):
            covariance[i, j] = np.mean(
                centered_variables[:, i] * centered_variables[:, j])

    return (means, covariance)

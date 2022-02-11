#!/usr/bin/env python3
"""Defines `batch_norm`."""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalizes an unactivated output of a neural network using batch
    normalization.

    Z: A numpy.ndarray of shape (m, n) that should be normalized.
        m: The number of data points.
        n: The number of features in Z.
    gamma: A numpy.ndarray of shape (1, n) containing the scales used for batch
        normalization.
    beta: A numpy.ndarray of shape (1, n) containing the offsets used for batch
        normalization.
    epsilon: A small number used to avoid division by zero.

    Returns: the normalized Z matrix
    """
    # Normalize Z
    mu = np.mean(Z, axis=0)
    variance = np.var(Z, axis=0)
    Z_norm = (Z - mu) / (variance + epsilon) ** .5
    # Parameterize the mean and standard deviation of Z
    return Z_norm * gamma + beta

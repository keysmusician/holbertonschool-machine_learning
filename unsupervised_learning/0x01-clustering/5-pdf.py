#!/usr/bin/env python3
""" Defines `initialize`. """
import numpy as np


def pdf(X, m, S):
    """
    Calculates the probability density function of a Gaussian distribution.

    X: A numpy.ndarray of shape (n, d) containing the data points whose PDF
        should be evaluated.
    m: A numpy.ndarray of shape (d,) containing the mean of the distribution.
    S: A numpy.ndarray of shape (d, d) containing the covariance of the
        distribution.

    Returns: A numpy.ndarray of shape (n,) containing the PDF values for each
        data point. All values in P have a minimum value of 1e-300. Returns
        None on failure.
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return
    n, d = X.shape
    if type(m) is not np.ndarray or \
            len(m.shape) != 1 or \
            d != m.shape[0] or \
            type(S) is not np.ndarray or \
            S.shape != (d, d):
        return None

    tau = 2 * np.pi
    pdf = np.exp(
        (X - m) @
        np.linalg.inv(S) @
        (X - m).T / -2) / \
        np.sqrt(tau ** d * np.linalg.det(S))

    tolerance = 1e-300
    return np.extract(pdf * np.identity(n) > tolerance, pdf)

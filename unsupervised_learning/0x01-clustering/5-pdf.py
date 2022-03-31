#!/usr/bin/env python3
""" Defines `pdf`. """
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
    if (
            type(m) is not np.ndarray or
            len(m.shape) != 1 or
            m.shape[0] != d or
            type(S) is not np.ndarray or
            S.shape != (d, d)
            ):
        return

    tau = 2 * np.pi
    normalization_constant = np.sqrt(tau ** d * np.linalg.det(S))
    X_centered_transpose = (X - m).T
    pdf = (
        np.exp(
            np.sum(
                np.linalg.inv(S) @
                X_centered_transpose *
                X_centered_transpose,
                axis=0
            ) / -2
        ) / normalization_constant
    )

    tolerance = 1e-300

    return np.where(pdf < tolerance, tolerance, pdf)

#!/usr/bin/env python3
"""Defines `GaussianProcess`."""
import numpy as np


class GaussianProcess:
    """A noiseless 1D Gaussian process."""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        X_init: A numpy.ndarray of shape (t, 1) representing the inputs already
            sampled with the black-box function.
            t: The number of initial samples.
        Y_init: A numpy.ndarray of shape (t, 1) representing the outputs of the
            black-box function for each input in X_init.
        l: The length parameter for the kernel.
        sigma_f: The standard deviation given to the output of the black-box
            function.
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """
        Calculates the covariance kernel matrix between two matrices using a
        Gaussian/squared exponential/Radial Basis Function (RBF) kernel.

        X1: A numpy.ndarray of shape (m, 1).
        X2: A numpy.ndarray of shape (n, 1).

        Returns: the covariance kernel matrix as a numpy.ndarray of shape
            (m, n).
        """
        sqdist = (
            np.sum(X1**2, 1).reshape(-1, 1) +
            np.sum(X2**2, 1) -
            2 * np.dot(X1, X2.T)
        )
        return self.sigma_f**2 * np.exp(-0.5 / self.l**2 * sqdist)

    def predict(self, X_s):
        """
        Predicts the mean and standard deviation of points in a Gaussian
        process.

        X_s: A numpy.ndarray of shape (s, 1) containing all of the points whose
            mean and standard deviation should be calculated.
            s: The number of sample points.

        Returns: (mu, sigma)
            mu: A numpy.ndarray of shape (s,) containing the mean for each
                point in X_s, respectively.
            sigma: A numpy.ndarray of shape (s,) containing the variance for
                each point in X_s, respectively.
        """
        K_s = self.kernel(self.X, X_s)
        K_sT_K_inv = K_s.T @ np.linalg.inv(self.K)
        mu = K_sT_K_inv @ self.Y
        sigma = self.kernel(X_s, X_s) - K_sT_K_inv @ K_s

        return (mu.flatten(), np.diag(sigma))


#!/usr/bin/env python3
"""Defines `mean_cov`."""
import numpy as np


class MultiNormal:
    """A multivariate normal distribution."""

    def __init__(self, data):
        """
        Initializes a MultiNormal.

        data: A numpy.ndarray of shape (d, n) containing the data set:
            d: The number of dimensions in each data point
            n: The number of data points
        """
        if type(data) != np.ndarray or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        elif data.shape[1] < 2:
            raise ValueError('data must contain multiple data points')

        sample_count = data.shape[1]
        self.mean = np.mean(data, axis=1, keepdims=True)
        mu = self.mean
        self.cov = np.matmul(data - mu, data.T - mu.T) / (sample_count - 1)

    def pdf(self, x):
        """
        Calculates the probability density at a data point.

        x: A numpy.ndarray of shape (d, 1) containing the data point whose PDF
            should be calculated:
            d: The number of dimensions of the MultiNormal instance

        Returns: The value of the PDF at a point.
        """
        if type(x) != np.ndarray:
            raise TypeError("x must be a numpy.ndarray")
        d = self.cov.shape[0]
        if len(x.shape) != 2:
            raise ValueError("x must have the shape ({}, 1)".format(d))
        d0, d1 = x.shape
        if d0 != d or d1 != 1:
            raise ValueError("x must have the shape ({}, 1)".format(d))

        tau = np.pi * 2
        probability_density = np.exp(
                -1/2 *
                (x - self.mean).T @
                np.linalg.inv(self.cov) @
                (x - self.mean)
            ) / np.sqrt(tau ** d * np.linalg.det(self.cov))

        return np.asscalar(probability_density)

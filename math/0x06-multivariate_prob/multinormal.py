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
            """"""

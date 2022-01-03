#!/usr/bin/env python3
"""Defines Neuron."""
import numpy as np


class Neuron:
    """A neural network neuron."""

    def __init__(self, nx):
        """
        Initializes a neuron.

        Args:
            nx: The number of input features to the neuron.
        """
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        elif nx < 1:
            raise ValueError('nx must be a positive integer')

        # A matrix containing a single row, which is the weights vector of the
        # neuron. Initialized with random weights:
        self.W = np.random.randn(1, nx)
        # The bias of the neuron:
        self.b = 0
        # The output/prediction of the neuron:
        self.A = 0


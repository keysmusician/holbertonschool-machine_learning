#!/usr/bin/env python3
"""Defines Neuron."""
import numpy as np


class Neuron:
    """An artificial neuron."""

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
        self.__W = np.random.randn(1, nx)
        # The bias of the neuron:
        self.__b = 0
        # The prediction/activation of the neuron:
        self.__A = 0

    @property
    def A(self):
        """The prediction/activation of the Neuron."""
        return self.__A

    @property
    def W(self):
        """The weights vector."""
        return self.__W

    @property
    def b(self):
        """The bias of the Neuron."""
        return self.__b

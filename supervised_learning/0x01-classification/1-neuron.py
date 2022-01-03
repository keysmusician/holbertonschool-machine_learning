#!/usr/bin/env python3
"""Defines Neuron."""
import numpy as np


class Neuron:
    """An artificial neuron."""

    @staticmethod
    def sigmoid(x):
        """The sigmoid function."""
        1 / (1 + np.e ** -x)

    activation_function = sigmoid

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

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the Neuron.

        Args:
            X: The input vector.
        """
        # activation = activation_function(input_vector * weight_vector + bias)
        self.__A = self.activation_function(np.dot(X, self.W) + self.b)

        return self.__A

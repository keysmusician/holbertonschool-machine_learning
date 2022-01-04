#!/usr/bin/env python3
"""Defines NeuralNetwork"""
import numpy as np


class NeuralNetwork:
    """A neural network."""

    def __init__(self, nx, nodes):
        """
        Initializes a NeuralNetwork.

        Args:
            nx: The number of input features.
            nodes: The number of nodes in the hidden layer.
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """The weights vector for the hidden layer."""
        return self.__W1

    @property
    def b1(self):
        """The bias vector for the hidden layer."""
        return self.__b1

    @property
    def A1(self):
        """The activation for the hidden layer."""
        return self.__A1

    @property
    def W2(self):
        """The weights vector for the output neuron."""
        return self.__W2

    @property
    def b2(self):
        """The bias of the output neuron."""
        return self.__b2

    @property
    def A2(self):
        """The activation of the output neuron (prediction)."""
        return self.__A2

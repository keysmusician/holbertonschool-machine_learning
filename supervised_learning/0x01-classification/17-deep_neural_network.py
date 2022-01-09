#!/usr/bin/env python3
"""Defines `DeepNeuralNetwork`."""
import numpy as np


class DeepNeuralNetwork:
    """
    A parameterized deep neural network.

        A frequently used parameter is X. X is the input to the Neuron. It is
    expected to be a matrix of type numpy.ndarray and shape (nx, m) where
    `nx` is the number of input features and `m` is the number of training
    examples. Each row contains the values of a single input feature for
    each training example, while each column contains the values of all
    input features for a single training example.
    """

    def __init__(self, nx, layers):
        """
        Initializes a deep neural network.

        Args:
            nx: The number of input features.
            layers: A list of the number of neurons in each layer.
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or not list:
            raise TypeError("layers must be a list of positive integers")

        # A dictionary of all weights and biases in the deep neural network:
        W_and_B = {}
        # The number of neurons in the previous layer; Used to improve random
        # initialization of weights:
        prev_neuron_count = nx
        layer_number = 1
        for neuron_count in layers:
            if type(neuron_count) is not int or neuron_count < 1:
                raise TypeError("layers must be a list of positive integers")
            weight_key = 'W{}'.format(layer_number)
            bias_key = 'b{}'.format(layer_number)

            W_and_B[bias_key] = np.zeros((neuron_count, 1))
            # He Normal initialization:
            W_and_B[weight_key] = \
                np.random.randn(neuron_count, prev_neuron_count) * \
                np.sqrt(2 / prev_neuron_count)

            prev_neuron_count = neuron_count
            layer_number += 1

        self.__weights = W_and_B
        self.__L = len(layers)
        self.__cache = {}

    @property
    def L(self):
        """The number of layers in the neural network."""
        return self.__L

    @property
    def weights(self):
        """The weights and biases of the neural network."""
        return self.__weights

    @property
    def cache(self):
        """A cache of intermediate values of the neural network."""
        return self.__cache

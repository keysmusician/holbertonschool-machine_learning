#!/usr/bin/env python3
"""Defines DeepNeuralNetwork"""
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
        W_and_B = dict()
        # The number of neurons in the previous layer; Used for better random
        # initialization of weights:
        prev_neuron_count = layers[-1]
        for layer_number, neuron_count in enumerate(layers, 1):
            if type(neuron_count) is not int:
                raise TypeError("layers must be a list of positive integers")
            weight_key = f'W{layer_number}'
            bias_key = f'b{layer_number}'
            # He Normal initialization:
            W_and_B[weight_key] = \
                np.random.randn(neuron_count, prev_neuron_count) * \
                np.sqrt(2/prev_neuron_count)
            W_and_B[bias_key] = np.zeros((neuron_count, 1))
            prev_neuron_count = neuron_count

        self.weights = W_and_B
        self.L = len(layers)
        self.cache = dict()

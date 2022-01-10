#!/usr/bin/env python3
"""Defines `DeepNeuralNetwork`."""
import numpy as np


def sigmoid(x):
    """The sigmoid function."""
    return 1 / (1 + np.e ** -x)


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
        if type(layers) is not list or not layers:
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

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network.

        Args:
            X: The training dataset.

        Returns: A tuple of:
            1) The output of the neural network, and
            2) The cache
        """
        activations = {"A0": X}
        for layer in range(1, self.L + 1):
            prev_layer_activation = activations["A{}".format(layer - 1)]

            activation = sigmoid(
                self.weights["W{}".format(layer)] @
                prev_layer_activation +
                self.weights["b{}".format(layer)]
            )
            activations["A{}".format(layer)] = activation

        self.__cache = activations
        return (activation, activations)

    def cost(self, Y, A):
        """
        The cost of the model.

        The cost is the average loss across the training dataset. Uses the
        cross-entropy loss function.

        Args:
            Y: The correct labels of the training dataset.
            A: The output of the neural network for each training example.

        Returns: The cost of the model.
        """
        # Using a value close to but not equal to 1 to prevent division by 0:
        _1 = 1.0000001
        log_loss = -(Y * np.log(A) + (1 - Y) * np.log(_1 - A))
        cost = np.mean(log_loss)
        return cost

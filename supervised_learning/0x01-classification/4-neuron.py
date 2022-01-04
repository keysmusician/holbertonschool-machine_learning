#!/usr/bin/env python3
"""Defines Neuron."""
import numpy as np


def sigmoid(x):
    """The sigmoid function."""
    return 1 / (1 + np.e ** -x)


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
        """
        The prediction/activation of the Neuron.

        A row vector of the activation of the Neuron for each training example.
        """
        return self.__A

    @property
    def W(self):
        """The weights row vector."""
        return self.__W

    @property
    def b(self):
        """The bias of the Neuron."""
        return self.__b

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the Neuron.

        Args:
            X: An input matrix of shape: (number of input features, number of
                training examples). Each row contains the values of a single
                input feature for each training example, while each column
                contains the values of all input features for a single training
                example.

            Returns:
                Neuron.A, a row vector of the activation of the Neuron for each
                training example.
        """
        # activation = activation_function(weight_vector * input_matrix + bias)
        Z = self.W @ X + self.b
        self.__A = sigmoid(Z)

        return self.__A

    def cost(self, Y, A):
        """
        The cost function of the Neural Network.

        Args:
            Y: A row vector of the correct labels for each training example.
            A: A row vector of the activation of a Neuron for each training
                example.
        """
        # Using a value close to but not equal to 1 to prevent division by 0:
        _1 = 1.0000001
        log_loss = -(Y * np.log(A) + (1 - Y) * np.log(_1 - A))
        cost = np.mean(log_loss)
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the Neuron's predictions.

        Args:
            X: An input matrix with shape (number of input features, number of
                training examples).
            Y: A row vector of the correct labels for each training example.

        Returns: Tuple of the Neuron's prediction and the cost of the neural
            network, respectively
        """
        activations = self.forward_prop(X)
        cost = self.cost(Y, activations)
        predictions = np.rint(activations).astype(int)
        return (predictions, cost)

#!/usr/bin/env python3
"""Defines NeuralNetwork"""
import numpy as np


def sigmoid(x):
    """The sigmoid function."""
    return 1 / (1 + np.e ** -x)


class NeuralNetwork:
    """
    A neural network with one hidden layer.

        A frequently used parameter is X. X is the input to the Neuron. It is
    expected to be a matrix of type numpy.ndarray and shape (nx, m) where
    `nx` is the number of input features and `m` is the number of training
    examples. Each row contains the values of a single input feature for
    each training example, while each column contains the values of all
    input features for a single training example.
    """

    def __init__(self, nx, nodes):
        """
        Initializes a NeuralNetwork.

        Args:
            nx: The number of input features.
            nodes: The number of neurons in the hidden layer.
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
        """The weights matrix for the hidden layer."""
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

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network.

        Args:
            X: The input matrix.

        Returns: A tuple of:
            1) the activation of each neuron in the the hidden layer for each
                training example, and
            2) the activation of the output neuron for each training example,
                respectively.
        """
        Z1 = self.W1 @ X + self.b1
        # Hidden layer activation:
        self.__A1 = sigmoid(Z1)
        Z2 = self.W2 @ self.__A1 + self.b2
        # Output activation:
        self.__A2 = sigmoid(Z2)
        return (self.__A1, self.__A2)

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression.

        The cost is the average loss across the training set.

        Args:
            Y: Correct labels for each training example in the input matrix.
            A: The activation of the output neuron for each training example.

        Returns: The cost of the model.
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
            X: Input matrix. See class description for further documentation.
            Y: A row vector of the correct labels for each training example.

        Returns: A tuple of:
            1) the neural network's predictions for each training example, and
            2) the cost of the neural network, respectively
        """
        output_activations = self.forward_prop(X)[1]
        predictions = np.rint(output_activations).astype(int)
        cost = self.cost(Y, output_activations)
        return (predictions, cost)

#!/usr/bin/env python3
"""Defines Neuron."""
import numpy as np


def sigmoid(x):
    """The sigmoid function."""
    return 1 / (1 + np.e ** -x)


class Neuron:
    """
    An artificial neuron.

    A frequently used parameter is X. X is the input to the Neuron. It is
        expected to be a matrix of type numpy.ndarray and shape (nx, m) where
        `nx` is the number of input features and `m` is the number of training
        examples. Each row contains the values of a single input feature for
        each training example, while each column contains the values of all
        input features for a single training example.
    """

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
        """The weights row vector for each input feature."""
        return self.__W

    @property
    def b(self):
        """The bias of the Neuron."""
        return self.__b

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the Neuron.

        Args:
            X: Input matrix. See class description for further documentation.

        Returns: Neuron.A, a row vector of the activation of the Neuron for
                each training example.
        """
        # activation = activation_function(weight_vector * input_matrix + bias)
        Z = self.W @ X + self.b
        self.__A = sigmoid(Z)

        return self.__A

    def cost(self, Y, A):
        """
        The cost function of the Neuron.

        Args:
            Y: A row vector of the correct labels for each training example.
            A: A row vector of the activation of the Neuron for each training
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
            X: Input matrix. See class description for further documentation.
            Y: A row vector of the correct labels for each training example.

        Returns: Tuple of the Neuron's prediction and the cost of the neural
            network, respectively
        """
        activations = self.forward_prop(X)
        cost = self.cost(Y, activations)
        predictions = np.rint(activations).astype(int)
        return (predictions, cost)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculates one step of gradient descent on the Neuron.

        Updates the weight vector and bias of the Neuron.

        Args:
            X: Input matrix. See class description for further documentation.
            Y: A row vector of the correct labels for the training examples.
            A: A row vector of the activated output of the neuron for each
                training example.
            alpha: The learning rate of the neural network.
        """
        # The derivatives of the loss function (l), with respect to the
        # linearly transformed inputs (Z) = dl_dZ; In logistic regression,
        # it simplifies to A - Y.
        dl_dZ = A - Y
        # The average (mean) derivatives of the loss function with respect to
        # each weight across the entire training dataset:
        dl_dW = X @ dl_dZ.T / X.shape[1]
        # Derivative of the loss function with respect to the bias:
        dl_db = dl_dZ
        self.__W -= alpha * dl_dW.T
        self.__b -= alpha * np.mean(dl_db)

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Trains the Neuron.

        Args:
            X: Input matrix. See class description for further documentation.
            Y: A row vector of the correct labels for the training examples.
            iterations: The number of times to train
            alpha: The learning rate
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")

        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")

        for iteration in range(iterations):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)

        return self.evaluate(X, Y)

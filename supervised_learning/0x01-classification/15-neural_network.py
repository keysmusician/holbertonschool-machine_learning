#!/usr/bin/env python3
"""Defines NeuralNetwork"""
import numpy as np
import matplotlib.pyplot as plt


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
        Initializes a neural network.

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
        """The activation matrix for the hidden layer."""
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
        """The activation vector of the output neuron (the prediction)."""
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
        Evaluates the neural network's predictions.

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

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Calculates one step of gradient descent on the neural network.

        Updates the weights and biases of the neural network.

        Args:
            X: The input matrix. See class description for further
                documentation.
            Y: A row vector of the correct labels for each training example.
            A1: A matrix of the activation of each neuron in the hidden layer
                for each training example.
            A2: A matrix of the activation of the output neuron for each
                training example.
            alpha: The learning rate of the neural network.
        """
        # The number of training examples:
        m = X.shape[1]

        # The derivatives of the loss function (l), with respect to the
        # linearly transformed inputs to the output neuron (Z2) = dl_dZ;
        # In logistic regression, it simplifies to A2 - Y:
        dl_dZ2 = A2 - Y

        # The average (mean) derivatives of the loss function with respect to
        # each of the output neuron's weights across the entire training
        # dataset:
        dl_dW2 = dl_dZ2 @ A1.T / m

        # Derivative of the loss function with respect to the output neuron's
        # bias:
        dl_db2 = np.mean(dl_dZ2, axis=1, keepdims=True)

        # Derivative of the loss function with respect to the hidden layer's
        # "z" values:
        dl_dZ1 = (self.W2.T @ dl_dZ2) * (A1 * (1 - A1))

        # The average (mean) of derivatives of the loss function with respect
        # to the hidden layer's weights:
        dl_dW1 = dl_dZ1 @ X.T / m

        # Derivative of the loss function with respect to the biases of the
        # hidden layer:
        dl_db1 = np.mean(dl_dZ1, axis=1, keepdims=True)

        self.__W1 -= alpha * dl_dW1
        self.__b1 -= alpha * dl_db1
        self.__W2 -= alpha * dl_dW2
        self.__b2 -= alpha * dl_db2

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """
        Trains the neural network.

        Args:
            X: Input matrix. See class description for further documentation.
            Y: A row vector of the correct labels for the training examples.
            iterations: The number of times to train.
            alpha: The learning rate.

        Returns: The evaluation of the training data after `iterations` cycles
            of training have occurred.
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")

        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")

        if verbose or graph:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step < 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
            x_iteration = []
            y_cost = []

        for iteration in range(iterations + 1):
            self.forward_prop(X)
            is_step_iteration = iteration % step == 0

            if is_step_iteration:
                if graph:
                    cost = self.cost(Y, self.A2)
                    x_iteration.append(iteration)
                    y_cost.append(cost)
                if verbose:
                    print(f"Cost after {iteration} iterations: {cost}")

            self.gradient_descent(X, Y, self.A1, self.A2, alpha)

        if graph:
            plt.plot(x_iteration, y_cost)
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)

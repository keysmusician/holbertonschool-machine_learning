#!/usr/bin/env python3
"""Defines `DeepNeuralNetwork`."""
import numpy as np
import matplotlib.pyplot as plt
import pickle


def sigmoid(x):
    """The sigmoid function."""
    return 1 / (1 + np.exp(-x))

def tanh(x):
    """The hyperbolic tangent function."""
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def one_hot_encode(Y, classes):
    """
    Converts a numeric label vector into a one-hot matrix.

    Args:
        Y: The correct labels of the training data.
        classes: The number of classes of the training data.

    Returns: A one-hot encoding of Y with shape (classes, m), or None on
        failure.
    """
    if (type(classes) is int and
            classes >= 2 and
            classes > max(Y)):
        return np.identity(classes)[Y].T


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

    def __init__(self, nx, layers, activation='sig'):
        """
        Initializes a deep neural network.

        Args:
            nx: The number of input features.
            layers: A list of the number of neurons in each layer.
            activation: A string representing the activation function used in
                the hidden layers.
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or not list:
            raise TypeError("layers must be a list of positive integers")
        if activation not in ('sig', 'tanh'):
            raise ValueError("activation must be 'sig' or 'tanh'")

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

        self.__activation = activation
        self.__cache = {}
        self.__weights = W_and_B
        self.__L = len(layers)

    @property
    def activation(self):
        """The name of the activation function used in the hidden layers."""
        return self.__activation

    @property
    def activation_function(self):
        """The activation function used in the hidden layers."""
        activation_functions = {'sig': sigmoid, 'tanh': tanh}
        return activation_functions[self.activation]

    @property
    def cache(self):
        """A cache of intermediate values of the neural network."""
        return self.__cache

    @property
    def L(self):
        """The number of layers in the neural network."""
        return self.__L

    @property
    def weights(self):
        """The weights and biases of the neural network."""
        return self.__weights

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network.

        Args:
            X: The training dataset.

        Returns: A tuple of:
            1) The output of the neural network, and
            2) The activations cache.
        """
        activations = {'A0': X}
        for layer in range(1, self.L + 1):
            prev_layer_activation = activations['A{}'.format(layer - 1)]

            activation = self.activation_function(
                self.weights['W{}'.format(layer)] @
                prev_layer_activation +
                self.weights['b{}'.format(layer)]
            )
            activations['A{}'.format(layer)] = activation

            # Make the output layer a softmax layer:
            if layer == self.L:
                t = np.exp(activation)
                activation = t / np.sum(t, axis=0, keepdims=True)

        self.__cache = activations
        return (activation, activations)

    def cost(self, Y, A):
        """
        The cost of the neural network.

        The cost is the average loss across the training dataset. Uses the
        cross-entropy loss function.

        Args:
            Y: The correct labels of the training dataset.
            A: The output of the neural network for each training example.

        Returns: The cost of the neural network.
        """
        cross_entropy = -np.sum(Y * np.log(A), axis=0)
        cost = np.mean(cross_entropy)
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network's predictions.

        Args:
            X: Training dataset. See class description for further
                documentation.
            Y: The correct labels of the training dataset.

        Returns: A tuple of:
            1) A one-hot matrix of the neural network's prediction, and
            2) The cost of the neural network.
        """
        output = self.forward_prop(X)[0]
        cost = self.cost(Y, output)
        prediction = np.eye(output.shape[0])[np.argmax(output, axis=0)].T
        return (prediction, cost)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one step of gradient descent on the neural network.

        Updates the weights and biases of the neural network.

        Args:
            Y: The correct labels of the training dataset.
            cache: A dictionary containing the activations of each layer of the
                neural network, where layer 0 is the training dataset.
            alpha: The learning rate of the neural network.
        """
        # The number of training examples:
        m = len(Y[0])

        # Activation of the final layer, aka the output:
        A_final = cache['A{}'.format(self.L)]

        dzh = A_final - Y

        for layer in range(self.L, 0, -1):

            # Activation of the previous layer:
            A_prev = cache['A{}'.format(layer - 1)]

            dwh = dzh @ A_prev.T / m

            dbh = np.mean(dzh, axis=1, keepdims=True)

            # Weights of the current layer:
            A = self.weights['W{}'.format(layer)]

            if self.activation == 'sig':
                dzl = A.T @ dzh * A_prev * (1 - A_prev)
            elif self.activation == 'tanh':
                dzl = A.T @ dzh * ( 1 - np.square(A_prev))

            # Bias of the current layer:
            b = self.weights['b{}'.format(layer)]

            A -= alpha * dwh
            b -= alpha * dbh
            dzh = dzl

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """
        Trains the Neuron.

        Args:
            X: Input matrix. See class description for further documentation.
            Y: A row vector of the correct labels for the training examples.
            iterations: The number of times to train.
            alpha: The learning rate.
            verbose: Whether or not to print information about the training.
            graph: Whether or not to graph information about the training once
                the training has completed.
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
            x = []
            y = []

        for iteration in range(iterations + 1):
            prediction = self.forward_prop(X)[0]
            is_step_iteration = iteration % step == 0

            if is_step_iteration:
                cost = self.cost(Y, prediction)
                if graph:
                    x.append(iteration)
                    y.append(cost)
                if verbose:
                    print('Cost after {} iterations: {}'
                          .format(iteration, cost))

            self.gradient_descent(Y, self.cache, alpha)

        if graph:
            plt.plot(x, y)
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """
        Saves a neural network instance to a file in pickle format.

        Args:
            filename: The name of the file to which the object should be saved.
        """
        if filename[-4:] != ".pkl":
            filename += ".pkl"
        with open(filename, "w+b") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        """
        Loads a pickled `DeepNeuralNetwork` object.

        Args:
            filename: The name of the file from which the object should be
                loaded.

        Returns: The loaded object, or None if filename doesn't exist.
        """
        try:
            with open(filename, "rb") as file:
                DeepNeuralNetwork_instance = pickle.load(file)
            return DeepNeuralNetwork_instance
        except FileNotFoundError:
            return None

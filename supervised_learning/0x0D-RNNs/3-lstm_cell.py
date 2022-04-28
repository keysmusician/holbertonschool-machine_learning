#!/usr/bin/env python3
""" Defines `LSTMCell`. """
import numpy as np


def sigmoid(x):
    """ Sigmoid activation function. """
    return 1 / (1 + np.power(np.e, -x))


def softmax(z):
    """ The softmax activation function. """
    a = np.exp(z)
    result = np.exp(z) / np.sum(a, axis=1, keepdims=True)
    return result


class LSTMCell:
    """ Represents a long short term memory cell. """

    def __init__(self, i, h, o):
        """
        Initializes an LSTMCell.

        i: The number of input data features.
        h: The number of hidden state features.
        o: The number of output features.
        """
        # Weights and biases for the...
        # Forget gate:
        self.Wf = np.random.randn(h + i, h)
        self.bf = np.zeros((1, h))
        # Update gate:
        self.Wu = np.random.randn(h + i, h)
        self.bu = np.zeros((1, h))
        # Cell state
        self.Wc = np.random.randn(h + i, h)
        self.bc = np.zeros((1, h))
        # Output gate:
        self.Wo = np.random.randn(h + i, h)
        self.bo = np.zeros((1, h))
        # Output features:
        self.Wy = np.random.randn(h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """
        Performs forward propagation for one time step.

        The output layer uses a softmax activation function.

        x_t: A numpy.ndarray of shape (batch_size, input_feature_count) that
            contains the data input for the cell.
        h_prev: A numpy.ndarray of shape (batch_size, hidden_feature_count)
            containing the previous hidden state.
        c_prev: A numpy.ndarray of shape (batch_size, hidden_feature_count)
            containing the previous cell state.

        Returns: (h_next, c_next, y):
            h_next: The next hidden state.
            c_next: The next cell state.
            y: The output of the cell.
        """
        cell_input = np.concatenate((h_prev, x_t), 1)
        forget_gate = sigmoid(cell_input @ self.Wf + self.bf)
        input_gate = sigmoid(cell_input @ self.Wu + self.bu)
        candidate_cell_state = np.tanh(cell_input @ self.Wc + self.bc)
        c_next = forget_gate * c_prev + input_gate * candidate_cell_state
        output_gate = sigmoid(cell_input @ self.Wo + self.bo)
        h_next = output_gate * np.tanh(c_next)
        y = softmax(h_next @ self.Wy + self.by)

        return (h_next, c_next, y)

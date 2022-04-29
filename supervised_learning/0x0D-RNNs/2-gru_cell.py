#!/usr/bin/env python3
""" Defines `GRUCell`. """
import numpy as np


def sigmoid(x):
    """ Sigmoid activation function. """
    return 1 / (1 + np.power(np.e, -x))


def softmax(z):
    """ The softmax activation function. """
    a = np.exp(z)
    result = np.exp(z) / np.sum(a, axis=1, keepdims=True)
    return result


class GRUCell:
    """ Represents a gated recurrent unit. """

    def __init__(self, i, h, o):
        """
        Initializes a GRU cell.

        i: The number of input data features.
        h: The number of hidden state features.
        o: The number of output features.
        """
        # Update gate weights and biases
        self.Wz = np.random.randn(i + h, h)
        self.bz = np.zeros((1, h))

        # Reset gate weights and biases
        self.Wr = np.random.randn(i + h, h)
        self.br = np.zeros((1, h))

        # Hidden state weights and biases
        self.Wh = np.random.randn(i + h, h)
        self.bh = np.zeros((1, h))

        # Output layer weights and biases
        self.Wy = np.random.randn(h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step.

        h_prev: A numpy.ndarray of shape (batch_size, hidden_feature_count)
            containing the previous hidden state.
        x_t: A numpy.ndarray of shape (batch_size, input_feature_count) that
            contains the input for the cell.

        Returns: (h_next, y):
            h_next: The next hidden state.
            output_features: The output features of the cell.
        """
        cell_input = np.concatenate((h_prev, x_t), axis=1)
        update_gate = sigmoid(cell_input @ self.Wz + self.bz)
        reset_gate = sigmoid(cell_input @ self.Wr + self.br)
        candidate_input = np.concatenate((reset_gate * h_prev, x_t), axis=1)
        candidate_state = np.tanh(candidate_input @ self.Wh + self.bh)
        h_next = update_gate * candidate_state + (1 - update_gate) * h_prev
        output_features = softmax(h_next @ self.Wy + self.by)

        return (h_next, output_features)

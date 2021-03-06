#!/usr/bin/env python3
""" Defines `RNNCell`. """
import numpy as np


def softmax(z):
    """ The softmax activation function. """
    a = np.exp(z)
    result = np.exp(z) / np.sum(a, axis=1, keepdims=True)
    return result


class RNNCell:
    """ Represents a cell of a simple RNN. """

    def __init__(self, i, h, o):
        """
        Initializes an RNNCell.

        i: The number of input features.
        h: The number of hidden features.
        o: The number of output features.
        """
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step.

        Uses a softmax activation function.

        h_prev: A numpy.ndarray of shape (batch_size, hidden_feature_count)
            containing the previous hidden state.
        x_t: T numpy.ndarray of shape (batch_size, input_feature_count) that
            contains the input data for the cell.

        Returns: (h_next, y):
            h_next: The next hidden state of shape (batch_size,
                hidden_feature_count).
            y: The output of the cell of shape (batch_size,
                output_feature_count).
        """
        # compute the new hidden state
        h_next = np.concatenate((h_prev, x_t), axis=1) @ self.Wh + self.bh
        h_next = np.tanh(h_next)

        # compute the output
        y = softmax(h_next @ self.Wy + self.by)

        return (h_next, y)

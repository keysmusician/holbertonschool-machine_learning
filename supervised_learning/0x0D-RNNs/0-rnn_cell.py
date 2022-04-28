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

        x_t: T numpy.ndarray of shape (batches, steps) that contains the data
            input for the cell.
        h_prev: A numpy.ndarray of shape (batches, hidden_features)
            containing the previous hidden state.

        Returns: (h_next, y):
            h_next: The next hidden state of shape (batch, hidden_features).
            y: The output of the cell of shape (batch, output_features).
        """
        # compute the new hidden state
        h_next = np.concatenate((h_prev, x_t), axis=1) @ self.Wh + self.bh
        h_next = np.tanh(h_next)

        # compute the output
        y = softmax(h_next @ self.Wy + self.by)

        return (h_next, y)

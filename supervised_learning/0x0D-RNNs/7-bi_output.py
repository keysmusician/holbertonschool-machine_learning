#!/usr/bin/env python3
""" Defines `BidirectionalCell`. """
import numpy as np


class BidirectionalCell:
    """ Represents a bidirectional cell. """

    def __init__(self, i, h, o):
        """
        Initializes a BidirectionalCell cell.

        i: The number of input data features.
        h: The number of hidden state features.
        o: The number of output features.
        """
        # Weights and biases for the...
        # Forward direction
        self.Whf = np.random.randn(i + h, h)
        self.bhf = np.zeros((1, h))
        # Backward direction
        self.Whb = np.random.randn(i + h, h)
        self.bhb = np.zeros((1, h))
        # Outputs
        self.Wy = np.random.randn(2 * h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Calculates the hidden state in the forward direction for one time step.

        h_prev: A numpy.ndarray of shape (batch_size, hidden_feature_count)
            containing the batch of previous hidden states.
        x_t: A numpy.ndarray of shape (batch_size, input_feature_count) that
            contains the input batch for the cell.

        Returns: (h_next, y):
            h_next: The next hidden state.
        """
        cell_input = np.concatenate((h_prev, x_t), 1)
        h_forward = np.tanh(cell_input @ self.Whf + self.bhf)

        return h_forward

    def backward(self, h_next, x_t):
        """
        Calculates the hidden state in the backward direction for one time
        step.

        h_next: A numpy.ndarray of shape (batch_size, hidden_feature_count)
            containing the batch of the next hidden states.
        x_t: A numpy.ndarray of shape (batch_size, input_feature_count) that
            contains the input batch for the cell.

        Returns: (h_prev):
            h_prev: The previous hidden state.
        """
        cell_input = np.concatenate((h_next, x_t), 1)
        h_backward = np.tanh(cell_input @ self.Whb + self.bhb)

        return h_backward

    def output(self, H):
        """
        Calculates all outputs for the RNN.

        H: A numpy.ndarray of shape (t, m, 2 * h) that contains the
            concatenated hidden states from both directions, excluding their
            initialized states.
            t: The number of time steps.
            m: The batch size for the data.
            h: The number of features in the hidden state.

        Returns: Y, the outputs.
        """
        Y = H @ self.Wy + self.by
        # softmax activation:
        Y = np.exp(Y) / np.sum(np.exp(Y), axis=2, keepdims=True)
        return Y

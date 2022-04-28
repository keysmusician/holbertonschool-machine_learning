#!/usr/bin/env python3
""" Defines `rnn`. """
import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Performs forward propagation for a simple RNN.

    rnn_cell: An instance of RNNCell that will be used for the forward
        propagation.
    X: The data to be used, given as a numpy.ndarray of shape (t, m, i).
        t: The maximum number of time steps.
        m: The batch size.
        i: The dimensionality of the data.
    h_0: The initial hidden state, given as a numpy.ndarray of shape (m, h).
        h: The dimensionality of the hidden state.

    Returns: (H, Y):
        H: A numpy.ndarray containing all of the hidden states.
        Y: A numpy.ndarray containing all of the outputs.
    """
    Y = []
    H = [h_0]
    hidden_state = h_0
    for step in X:
        hidden_state, output = rnn_cell.forward(hidden_state, step)
        H.append(hidden_state)
        Y.append(output)

    return (np.array(H), np.array(Y))

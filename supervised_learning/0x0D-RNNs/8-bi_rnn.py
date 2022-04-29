#!/usr/bin/env python3
""" Defines `bi_rnn`. """
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    Performs forward propagation for a bidirectional RNN.

    bi_cell: An instance of BidirectinalCell that will be used for forward
        propagation.
    X: The data to be used, given as a numpy.ndarray of shape (t, m, i).
        t: The total number of steps.
        m: The batch size.
        i: The number of features in the input data.
    h_0: The initial hidden state in the forward direction, given as a
        numpy.ndarray of shape (m, h).
        h: The number of hidden features.
    h_t: The initial hidden state in the backward direction, given as a
        numpy.ndarray of shape (m, h).

    Returns: (H, Y)
        H: A numpy.ndarray containing all of the concatenated hidden states.
        Y: A numpy.ndarray containing all of the outputs.
    """
    h_next = h_0
    h_prev = h_t
    forward_hidden_states = []
    backward_hidden_states = []

    for step in range(len(X)):
        h_next = bi_cell.forward(h_next, X[step])
        forward_hidden_states.append(h_next)

        reverse_step = -(step + 1)
        h_prev = bi_cell.backward(h_prev, X[reverse_step])
        backward_hidden_states.append(h_prev)

    hidden_states = []
    for forward_hidden_state, backward_hidden_state in \
            zip(forward_hidden_states, reversed(backward_hidden_states)):
        merged_hidden_state = np.concatenate(
            (forward_hidden_state, backward_hidden_state), 1)
        hidden_states.append(merged_hidden_state)

    return (np.array(hidden_states), bi_cell.output(hidden_states))

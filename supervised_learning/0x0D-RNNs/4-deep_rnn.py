#!/usr/bin/env python3
""" Defines `deep_rnn`. """
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Performs forward propagation for a simple RNN.

    rnn_cells: A list of RNNCell instances of length `l` that will be used for
        the forward propagation.
        l: The number of layers.
    X: The data to be used, given as a numpy.ndarray of shape (t, m, i).
        t: The maximum number of steps.
        m: The batch size.
        i: The dimensionality of the data.
    h_0: The initial hidden states for each RNN cell, given as a numpy.ndarray
        of shape (l, m, h).
        h: The number of features in the hidden state.

    Returns: (H, Y):
        H: A numpy.ndarray containing the all of the hidden states; The batch
            of inital hidden states, plus the batches of hidden states returned
            from each cell at each step. The shape is:
                (steps + 1, cells, batch_size, hidden_feature_count)
        Y: A numpy.ndarray containing the batch of outputs at each step. The
            shape is (steps, batch_size, output_feature_count).
            output_feature_count = the number of features in the final cell in
                `rnn_cells`.
    """
    previous_hidden_states = h_0
    all_hidden_states = [h_0]
    all_outputs = []
    for step in X:
        # `step` shape is (batch_size, input_feature_count).
        next_hidden_states = []
        next_hidden_state = step
        for previous_hidden_state, cell in \
                zip(previous_hidden_states, rnn_cells):
            next_hidden_state, y = cell.forward(
                previous_hidden_state, next_hidden_state)
            next_hidden_states.append(next_hidden_state)
        all_hidden_states.append(next_hidden_states)
        previous_hidden_states = next_hidden_states
        #  `y` shape is (batch_size, output_feature_count).
        all_outputs.append(y)

    return (np.array(all_hidden_states), np.array(all_outputs))

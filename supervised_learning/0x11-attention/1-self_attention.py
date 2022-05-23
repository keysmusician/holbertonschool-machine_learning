#!/usr/bin/env python3
""" Defines `SelfAttention`. """
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """ Calculates the attention for machine translation. """

    def __init__(self, units):
        """
        Initializes a SelfAttention layer.

        units: An integer representing the number of hidden units in the
            alignment model.
        """
        super().__init__()
        # For the previous decoder hidden state:
        self.W = tf.keras.layers.Dense(units)
        # For the encoder hidden states:
        self.U = tf.keras.layers.Dense(units)
        # For the tanh of the sum of the outputs of W and U:
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """
        Runs when a SelfAttention layer is called.

        s_prev: A tensor of shape (batch, units) containing the previous
            decoder hidden state.
        hidden_states: A tensor of shape (batch, input_seq_len, units)
            containing the outputs of the encoder.

        Returns: (context, weights)
            context: A tensor of shape (batch, units) that contains the context
                vector for the decoder.
            weights: A tensor of shape (batch, input_seq_len, 1) that contains
                the attention weights.
        """
        return self.V(tf.tanh(self.W(s_prev) + self.U(hidden_states)))

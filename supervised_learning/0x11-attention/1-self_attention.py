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
        hidden_states: Also called annotations, a tensor of shape (batch,
            input_seq_len, units) containing the outputs of the encoder.

        Returns: (context_vector, weights)
            context_vector: A tensor of shape (batch, units) that contains the
                context vector for the decoder.
            weights: A tensor of shape (batch, input_seq_len, 1) that contains
                the attention/alignment model weights.
        """
        # hidden states/annotations shape = (batch_size/sentence_count,
        # word/token_count, unit_count/latent_features)

        # alignment scores & weights shape =
        # (batch_size/sentence_count, 1 [expanded dimension], word/token_count)

        s_prev_expanded = tf.expand_dims(s_prev, 1)

        # The alignment model:
        alignment_scores = self.V(
            tf.nn.tanh(self.W(s_prev_expanded) + self.U(hidden_states)))
        alignment_weights = tf.nn.softmax(alignment_scores, axis=1)

        # Calculate the context vector (aka the expected annotation).
        # Swap alignment_weights axis 1 (added for broadcasting agreement) with
        # axis 2 (words/tokens), then matrix multiply. Finally, remove the
        # expanded dimension that was added for broadcasting:
        # context_vector = tf.squeeze(
        #   tf.transpose(alignment_weights, [0, 2, 1]) @ hidden_states
        # )

        # Simpler calculation:
        context_vector = tf.reduce_sum(
            alignment_weights * hidden_states, axis=1)

        return context_vector, alignment_weights

#!/usr/bin/env python3
""" Defines `MultiHeadAttention`. """
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """ A multi-head attention layer. """

    def __init__(self, dm, h):
        """
        Initializes a MultiHeadAttention layer.

        dm: An integer divisible by h representing the dimensionality of the
            model.
        h: An integer representing the number of heads.
        """
        super().__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def call(self, Q, K, V, mask):
        """
        Executes the MultiHeadAttention layer.

        Q: A tensor of shape (batch, seq_len_q, dk) containing the input to
            generate the query matrix.
            dk: The number of feature dimensions in `K`.
        K: A tensor of shape (batch, seq_len_v, dk) containing the input to
            generate the key matrix.
        V: A tensor of shape (batch, seq_len_v, dv) containing the input to
            generate the value matrix.
            dv: The number of feature dimensions in `V`.
        mask: Always None.

        Returns: (output, weights)
            output: A tensor with its last two dimensions as (..., seq_len_q,
                dm) containing the scaled dot product attention.
            weights: A tensor with its last three dimensions as (..., h,
                seq_len_q, seq_len_v) containing the attention weights.
        """
        attention_parameters = [
            self.Wq(Q),
            self.Wk(K),
            self.Wv(V)
        ]
        for i, parameter in enumerate(attention_parameters):
            # Split the feature axis into heads x depth, where depth is a
            # subset/slice of the features
            # Then, swap the heads & tokens axes
            attention_parameters[i] = tf.transpose(
                tf.reshape(
                    parameter, (*parameter.shape[:-1], self.h, self.depth)
                ),
                perm=[0, 2, 1, 3]
            )

        Q, K, V = attention_parameters
        attention_scores, weights = sdp_attention(Q, K, V, mask)
        # Un-swap the heads & tokens axes
        attention_scores = tf.transpose(attention_scores, perm=[0, 2, 1, 3])
        # And merge the heads back into a single features axis
        attention_scores = tf.reshape(
            attention_scores, (*attention_scores.shape[:-2], self.dm))
        output = self.linear(attention_scores)

        return output, weights

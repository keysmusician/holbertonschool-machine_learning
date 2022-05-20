#!/usr/bin/env python3
""" Defines `RNNEncoder`. """
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """ Encodes for machine translation. """

    def __init__(self, vocab, embedding, units, batch):
        """
        Initializes an RNN encoder.

        vocab: An integer representing the size of the input vocabulary.
        embedding: An integer representing the dimensionality of the embedding
            vector.
        units: An integer representing the number of hidden units in the RNN
            cell.
        batch: An integer representing the batch size.
        """
        super().__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True
        )

    def initialize_hidden_state(self):
        """
        Initializes the hidden states for the RNN cell to a tensor of zeros.

        Returns: A tensor of shape (batch, units) containing the initialized
            hidden states.
        """
        return tf.zeros([self.batch, self.units])

    def call(self, x, initial):
        """
        The operation run when an RNNEncoder layer is called.

        x: A tensor of shape (batch, input_seq_len) containing the input to the
            encoder layer as word indices within the vocabulary.
        initial: A tensor of shape (batch, units) containing the initial hidden
            state.

        Returns: (outputs, hidden)
            outputs: A tensor of shape (batch, input_seq_len, units) containing
                the outputs of the encoder.
            hidden: A tensor of shape (batch, units) containing the last hidden
                state of the encoder.
        """
        return self.gru(self.embedding(x), initial)

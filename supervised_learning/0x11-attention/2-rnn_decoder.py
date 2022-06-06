#!/usr/bin/env python3
""" Defines `RNNDecoder`. """
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """ An RNN decoder layer. """

    def __init__(self, vocab, embedding, units, batch):
        """
        Initializes an RNNDecoder.

        vocab: An integer representing the size of the output vocabulary.
        embedding: An integer representing the dimensionality of the embedding
            vector.
        units: An integer representing the number of hidden units in the RNN
            cell.
        batch: An integer representing the batch size.
        """
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform',
        )
        self.F = tf.keras.layers.Dense(vocab)

    def call(self, x, s_prev, hidden_states):
        """
        Executes the RNNDecoder layer.

        x: A tensor of shape (batch, 1) containing the previous word in the
            target sequence as an index of the target vocabulary.
        s_prev: A tensor of shape (batch, units) containing the previous
            decoder hidden state.
        hidden_states: A tensor of shape (batch, input_seq_len, units)
            containing the outputs of the encoder.

        Returns: (y, s)
            y: A tensor of shape (batch, vocab) containing the output word as a
                one hot vector in the target vocabulary.
            s: A tensor of shape (batch, units) containing the new decoder
                hidden state.
        """
        embeddings = self.embedding(x)
        context_vector, _ = SelfAttention(s_prev.shape[1])(
            s_prev, hidden_states)
        context_vector = tf.expand_dims(context_vector, 1)
        concat_input = tf.concat([context_vector, embeddings], -1)
        output, hidden_state = self.gru(concat_input)
        word_embedding = self.F(tf.reshape(output, (-1, output.shape[2])))

        return word_embedding, hidden_state

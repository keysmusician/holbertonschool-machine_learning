#!/usr/bin/env python3
""" Defines `RNNDecoder`. """
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """ An RNN decoder layer. """

    def __init__(self, vocab, embedding, units, batch):
        """ Initializes an RNNDecoder. """
        self.embedding = tf.keras.layers.Embedding(embedding, vocab)
        self.gru = tf.keras.layers.GRU(
            units, return_sequences=True, return_state=True)
        self.F = tf.keras.layers.Dense(units)

    def call(self, x, s_prev, hidden_states):
        """ Call """
        pass

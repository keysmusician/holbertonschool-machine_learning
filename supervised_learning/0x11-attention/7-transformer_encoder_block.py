#!/usr/bin/env python3
""" Defines `EncoderBlock`. """
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """ An encoder block layer. """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """ Initializes a EncoderBlock layer. """
        self.mha = MultiHeadAttention()
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, Q, K, V, mask):
        """ Call """
        pass

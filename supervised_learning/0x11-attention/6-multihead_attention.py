#!/usr/bin/env python3
""" Defines `MultiHeadAttention`. """
import tensorflow as tf


class MultiHeadAttention(tf.keras.layers.Layer):
    """ A multi-head attention layer. """

    def __init__(self, dm, h):
        """ Initializes a MultiHeadAttention layer. """
        self.h = h
        self.dm = dm
        self.depth = dm / h
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def call(self, Q, K, V, mask):
        """ Call """
        pass

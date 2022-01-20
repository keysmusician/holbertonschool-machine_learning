#!/usr/bin/env python3
"""Defines `dropout_create_layer`."""
import tensorflow.compat.v1 as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """
    Creates a layer of a neural network using dropout.

    prev: A tensor containing the output of the previous layer.
    n: The number of nodes the new layer should contain.
    activation: The activation function that should be used on the layer.
    keep_prob: The probability that a node will be kept.

    Returns: The output of the new layer.
    """
    dense = tf.layers.Dense(n, activation)
    dropout = tf.layers.Dropout(keep_prob)
    return dropout(dense(prev))

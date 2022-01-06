#!/usr/bin/env python3
"""Defines `create_placeholders`."""
import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """
    Creates a neural network layer.

    Args:
        prev: The tensor output of the previous layer.
        n: The number of neurons desired in the layer.
        activation: The activation function that the layer should use.

    Returns: The tensor output of the layer.
    """
    layer = tf.layers.Dense(n, activation, name="layer")
    return layer(prev)

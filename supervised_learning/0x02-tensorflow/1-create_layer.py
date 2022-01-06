#!/usr/bin/env python3
"""Defines `create_placeholders`."""
import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """
    Creates a neural network layer.

    Initializes weights with He Normal initialization.

    Args:
        prev: The tensor output of the previous layer.
        n: The number of neurons desired in the layer.
        activation: The activation function that the layer should use.

    Returns: The tensor output of the layer.
    """
    layer = tf.layers.Dense(
        activation=activation,
        name="layer",
        units=n,
        kernel_initializer=tf.keras.initializers.
            VarianceScaling(mode='fan_avg')
    )
    return layer(prev)

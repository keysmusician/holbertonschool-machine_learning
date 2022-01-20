#!/usr/bin/env python3
"""Defines `l2_reg_create_layer`."""
import tensorflow.compat.v1 as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a tensorflow layer that includes L2 regularization.

    prev: A tensor containing the output of the previous layer.
    n: The number of nodes the new layer should contain.
    activation: The activation function that should be used on the layer.
    lambtha: The L2 regularization parameter.

    Returns: The output of the new layer.
    """
    L2 = tf.keras.regularizers.L2(lambtha)
    return tf.keras.layers.Dense(n, activation, kernel_regularizer=L2)(prev)

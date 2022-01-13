#!/usr/bin/env python3
"""Defines `create_batch_norm_layer`."""
import tensorflow.compat.v1 as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in tensorflow.

    prev: The activated output of the previous layer.
    n: The number of nodes in the layer to be created.
    activation: The activation function that should be used on the output of
        the layer.

    Returns: a tensor of the activated output for the layer
    """
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.layers.Dense(n, kernel_initializer=init)
    new = layer(prev)

    mean, var = tf.nn.moments(new, axes=[0])
    beta = tf.Variable(
        tf.constant(0.0, shape=[n]), trainable=True, name='beta')
    gamma = tf.Variable(
        tf.constant(1.0, shape=[n]), trainable=True, name='gamma')
    epsilon = 1e-8
    batch_normalization = tf.nn.batch_normalization(
        new, mean, var, beta, gamma, epsilon)

    return activation(batch_normalization)

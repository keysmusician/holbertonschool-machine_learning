#!/usr/bin/env python3
"""Defines `create_Adam_op`."""
import tensorflow.compat.v1 as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    Creates the training operation for a neural network in tensorflow using the
    Adam optimization algorithm.

    loss: The loss of the network.
    alpha: The learning rate.
    beta1: The weight used for the first moment.
    beta2: The weight used for the second moment.
    epsilon: A small number to avoid division by zero.

    Returns: the Adam optimization operation.
    """
    adam = tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon)
    return adam.minimize(loss)

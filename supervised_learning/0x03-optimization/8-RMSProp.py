#!/usr/bin/env python3
"""Defines `create_RMSProp_op`."""
import tensorflow.compat.v1 as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
    Creates the training operation for a neural network in tensorflow using the
    RMSProp optimization algorithm.

    loss: The loss of the network.
    alpha: The learning rate.
    beta2: The RMSProp weight.
    epsilon: A small number to avoid division by zero.

    Returns: The RMSProp optimization operation.
    """
    RMS_prop = tf.train.RMSPropOptimizer(alpha, beta2, epsilon=epsilon)
    return RMS_prop.minimize(loss)

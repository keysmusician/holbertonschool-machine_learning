#!/usr/bin/env python3
"""Defines `update_variables_momentum`."""
import tensorflow.compat.v1 as tf


def create_momentum_op(loss, alpha, beta1):
    """
    Creates the training operation for a neural network in tensorflow using the
    gradient descent with momentum optimization algorithm.

    loss: The loss of the network.
    alpha: The learning rate.
    beta1: The momentum weight.

    Returns: the momentum optimization operation.
    """
    optimizer = tf.train.MomentumOptimizer(alpha, beta1)
    return optimizer.minimize(loss)

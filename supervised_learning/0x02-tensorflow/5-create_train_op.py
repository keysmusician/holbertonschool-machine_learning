#!/usr/bin/env python3
"""Defines `create_train_op`."""
import tensorflow.compat.v1 as tf


def create_train_op(loss, alpha):
    """
    Creates the training operation for the network.

    Args:
        loss: The loss of the network's prediction.
        alpha: The learning rate.

    Returns: An operation that trains the network using gradient descent.
    """
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    train = optimizer.minimize(loss)
    return train

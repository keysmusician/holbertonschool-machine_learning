#!/usr/bin/env python3
"""Defines `create_placeholders`."""
import tensorflow.compat.v1 as tf


def create_placeholders(nx, classes):
    """
    Returns two placeholders, x and y, for the neural network.

    Args:
        nx: The number of feature columns in our data.
        classes: The number of classes in our classifier.

    Returns: A tuple of:
        1) The placeholder for the input data to the neural network (x), and
        2) The placeholder for the one-hot labels for the input data (y),
        respectively
    """
    x = tf.placeholder(name='x', dtype=tf.float32, shape=(None, nx))
    y = tf.placeholder(name='y', dtype=tf.float32, shape=(None, classes))
    return (x, y)

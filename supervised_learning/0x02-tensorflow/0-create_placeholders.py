#!/usr/bin/env python3
"""Defines `create_placeholders`."""
import tensorflow.compat.v1 as tf


def create_placeholders(nx, classes):
    """
    Returns two placeholders, x and y, for the neural network.

    Args:
        nx: The number of feature columns in our data.
        classes: The number of classes in our classifier.
    """
    return (x:=tf.placeholder(), y:=tf.placeholder)

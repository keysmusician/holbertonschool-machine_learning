#!/usr/bin/env python3
"""Defines `one_hot`."""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    Converts a label vector into a one-hot matrix.

    The last dimension of the one-hot matrix is the number of classes.

    Returns: The one-hot matrix.
    """
    return K.utils.to_categorical(labels, classes)

#!/usr/bin/env python3
"""Defines `one_hot_encode`."""
import numpy as np


def one_hot_encode(Y, classes):
    """
    Converts a numeric label vector into a one-hot matrix.

    Args:
        Y: The correct labels of the training data.
        classes: The number of classes of the training data.

    Returns: A one-hot encoding of Y with shape (classes, m), or None on
        failure.
    """
    if (type(classes) is int and
            classes >= 2 and
            classes > max(Y)):
        return np.identity(classes)[Y].T

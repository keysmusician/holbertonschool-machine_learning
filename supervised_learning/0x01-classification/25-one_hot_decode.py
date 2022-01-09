#!/usr/bin/env python3
"""Defines `one_hot_encode`."""
import numpy as np


def one_hot_decode(one_hot):
    """
    Converts a one-hot matrix into a numeric label vector.

    Args:
        Y: The correct labels of the training data.
        classes: The number of classes of the training data.

    Returns: A one-hot deencoding of Y with shape (classes, m), or None on
        failure.
    """
    try:
        return np.argmax(one_hot, axis=0)
    except Exception:
        return None

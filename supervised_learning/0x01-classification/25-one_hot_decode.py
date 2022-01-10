#!/usr/bin/env python3
"""Defines `one_hot_decode`."""
import numpy as np


def one_hot_decode(one_hot):
    """
    Converts a one-hot matrix into a numeric label vector.

    Args:
        one_hot: A one-hot matrix.

    Returns: A decoding of `one_hot` with shape (classes, traing examples), or
        None on failure.
    """
    if type(one_hot) is np.ndarray and one_hot.ndim == 2:
        return np.argmax(one_hot, axis=0)



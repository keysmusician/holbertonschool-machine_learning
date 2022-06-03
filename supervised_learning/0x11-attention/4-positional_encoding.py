#!/usr/bin/env python3
""" Defines `positional_encoding`. """
import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    Calculates the sinusoidal positional encoding for a transformer.

    max_seq_len: An integer representing the maximum sequence length.
    dm: The model depth.

    Returns: A `numpy.ndarray` of shape (max_seq_len, dm) containing the
        positional encoding vectors.
    """
    yy, xx = np.meshgrid(np.arange(dm), np.arange(max_seq_len))

    return np.sin(xx / 10000 ** (yy // 2 * 2 / dm) + np.pi / 2 * (yy % 2))

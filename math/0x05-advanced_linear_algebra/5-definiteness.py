#!/usr/bin/env python3
""" Defines `definiteness`. """
import numpy as np


def definiteness(matrix):
    """
    Calculates the definiteness of a matrix.

    matrix: A square numpy.ndarray.

    Returns: A string: Positive definite, Positive semi-definite, Negative
        semi-definite, Negative definite, or Indefinite.
    """
    if type(matrix) is not np.ndarray:
        raise TypeError('matrix must be a numpy.ndarray')
    elif not np.array_equal(matrix.T, matrix) or not matrix.any():
        return

    determinants = []
    for n in range(1, len(matrix) + 1):
        determinants.append(
            np.linalg.det(matrix[:n, :n])
        )

    # This solution is inefficient, but that's not a big deal in this case.
    if all([d > 0 for d in determinants]):
        return 'Positive definite'
    elif all([d >= 0 for d in determinants]):
        return 'Positive semi-definite'
    elif all([
            d < 0 if i % 2 == 0 else d > 0
            for i, d in enumerate(determinants)
            ]):
        return 'Negative definite'
    elif all([
            d <= 0 if i % 2 == 0 else d >= 0
            for i, d in enumerate(determinants)
            ]):
        return 'Negative semi-definite'
    else:
        return 'Indefinite'

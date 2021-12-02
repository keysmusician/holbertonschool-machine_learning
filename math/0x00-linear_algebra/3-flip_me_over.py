#!/usr/bin/env python3
"""Defines `matrix_transpose`."""


def matrix_transpose(matrix):
    """Transposes a 2D matrix."""
    transposed = [
        [row[n] for row in matrix]
        for n in range(len(matrix[0]))
    ]

    return transposed

#!/usr/bin/env python3
"""Defines `matrix_shape`."""


def matrix_shape(matrix):
    """Returns the dimensions of a matrix."""
    dimentions = []
    while True:
        try:
            iter(matrix)
            matrix[0]
        except (TypeError, IndexError):
            return dimentions
        dimentions.append(len(matrix))
        matrix = matrix[0]

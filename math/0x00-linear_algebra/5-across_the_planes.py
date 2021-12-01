#!/usr/bin/env python3
"""Defines `add_matrices2D`."""


def add_matrices2D(mat1, mat2):
    """Adds two matrices element-wise."""
    if len(mat1) != len(mat2):
        return None
    if len(mat1) == 0:
        return []
    if len(mat1[0]) != len(mat2[0]):
        return None

    matrix = list()
    for row, row_pair in enumerate(zip(mat1, mat2)):
        matrix.append([])
        for tuple in zip(row_pair[0], row_pair[1]):
            matrix[row].append(sum(tuple))

    return matrix

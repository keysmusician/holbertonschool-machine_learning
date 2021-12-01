#!/usr/bin/env python3
"""Defines `cat_matrices2D`."""


def cat_matrices2D(mat1, mat2, axis=0):
    """Concatenates two matrices along a specific axis."""
    mat1 = [row.copy() for row in mat1]
    mat2 = [row.copy() for row in mat2]
    if axis == 0 and len(mat1[0]) == len(mat2[0]):
        return mat1 + mat2
    elif axis == 1 and len(mat1) == len(mat2):
        return [row + mat2[index] for index, row in enumerate(mat1)]

#!/usr/bin/env python3
"""Defines `add_matrices2D`."""
from itertools import zip_longest


def add_matrices2D(mat1, mat2):
    """Adds two matrices element-wise."""
    try:
        return [sum(tuple) for rows in zip_longest(mat1, mat2)
                for tuple in zip_longest(*rows)]
    except TypeError:
        return None

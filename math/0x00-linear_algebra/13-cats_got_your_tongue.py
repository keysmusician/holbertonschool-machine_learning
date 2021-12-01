#!/usr/bin/env python3
"""Defines `np_cat`."""
import numpy


def np_cat(mat1, mat2, axis=0):
    """Concatenates two matrices along a specific axis."""
    return numpy.concatenate((mat1, mat2), axis)

#!/usr/bin/env python3
"""Defines `np_elementwise`."""


def np_elementwise(mat1, mat2):
    """
    Performs element-wise addition, subtraction, multiplication, and division.

    Returns: A tuple containing the element-wise sum, difference, product, and
        quotient, respectively
    """
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)

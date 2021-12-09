#!/usr/bin/env python3
"""Defines `poly_integral`."""


def poly_integral(poly, C=0):
    """
    Calculates the indefinite integral of a polynomial from a list of its
    coefficients.
    """
    if type(C) is not int:
        return None
    try:
        poly = [n / (i + 1) for i, n in enumerate(poly)]
        poly = [C] + poly
        poly = \
            [int(n) if type(n) is float and n.is_integer() else n for n in poly]
        return poly
    except TypeError:
        return None

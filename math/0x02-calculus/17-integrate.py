#!/usr/bin/env python3
"""Defines `poly_integral`."""


def poly_integral(poly, C=0):
    """
    Calculates the indefinite integral of a polynomial from a list of its
    coefficients.
    """
    if not poly or \
            type(poly) is not list or \
            type(C) not in (int, float) or \
            not all([type(n) in (int, float) for n in poly]):
        return None

    # Truncate trailing zeros:
    try:
        while poly[-1] == 0:
            del poly[-1]
    except IndexError:
        pass

    poly = [n / (i + 1) for i, n in enumerate(poly)]
    poly = [C] + poly

    # Convert all whole number floats to integers
    poly = [int(n) if type(n) is float and n.is_integer() else n
            for n in poly]

    return poly

#!/usr/bin/env python3
"""Defines `poly_derivative`."""


def poly_derivative(poly):
    """
    Calculates the derivative of a polynomial from a list of its coefficients.
    """
    if not poly or \
            type(poly) is not list or \
            not all([type(n) in (int, float) for n in poly]):
        return None

    derivative = [i * n for i, n in enumerate(poly)][1:]

    return derivative if derivative else [0]

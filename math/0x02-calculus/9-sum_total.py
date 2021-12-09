#!/usr/bin/env python3
"""Defines `summation_i_squared`."""


def summation_i_squared(n):
    """Calculates the sum of squares of numbers from 1 to `n`."""
    if isinstance(n, (int, float)):
        return sum(map(lambda i: i ** 2, range(1, n + 1)))

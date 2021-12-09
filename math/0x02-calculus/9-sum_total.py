"""Defines `summation_i_squared`."""


def summation_i_squared(n):
    """Calculates the sum of squares of numbers from 1 to n."""
    return sum([m ** 2 for m in range(1, n + 1)])

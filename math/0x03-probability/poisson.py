#!/usr/bin/env python3
"""Defines `Poisson`."""


class Poisson:
    """A Poisson distribution."""

    def __init__(self, data=None, lambtha=1.):
        """Initializes a Poisson distribution."""
        if data is None:
            if lambtha <= 0:
                raise ValueError('lambtha must be a positive value')
            self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError('data must be a list')
            elif len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.lambtha = sum(data) / len(data)

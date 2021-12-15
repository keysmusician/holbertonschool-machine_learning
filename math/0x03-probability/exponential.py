#!/usr/bin/env python3
"""Defines `Exponential`."""


class Exponential:
    """An exponential distribution."""

    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """Initializes an exponential distribution."""
        if data is None:
            if lambtha <= 0:
                raise ValueError('lambtha must be a positive value')
            self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError('data must be a list')
            elif len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.lambtha = 1 / (sum(data) / len(data))

    def pdf(self, x):
        """The probability density function."""
        if x < 0:
            return 0

        return self.lambtha * self.e ** (-self.lambtha * x)

    def cdf(self, x):
        """The cumulative distribution function."""
        if x < 0:
            return 0

        return 1 - self.e ** (-self.lambtha * x)

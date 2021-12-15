#!/usr/bin/env python3
"""Defines `Poisson`."""


class Poisson:
    """A Poisson distribution."""

    e = 2.7182818285

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

    def pmf(self, k):
        """The probability mass function."""
        def factorial(n):
            if n == 1 or n == 0:
                return 1
            else:
                return n * factorial(n-1)

        k = int(k)
        if k < 0:
            return 0

        return (self.lambtha ** k * self.e ** -self.lambtha) / factorial(k)

    def cdf(self, k):
        """The cumulative distribution function."""
        k = int(k)
        if k <= 0:
            return 0

        cdf = 0
        while k >= 0:
            cdf += self.pmf(k)
            k -= 1

        return cdf

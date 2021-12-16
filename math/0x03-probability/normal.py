#!/usr/bin/env python3
"""Defines `Normal`."""


class Normal:
    """A normal distribution."""

    e = 2.7182818285
    pi = 3.1415926536
    tau = 2 * pi

    def __init__(self, data=None, mean=0., stddev=1.):
        """Initializes a normal distribution."""
        if data is None:
            if stddev <= 0:
                raise ValueError('stddev must be a positive value')
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if type(data) is not list:
                raise TypeError('data must be a list')
            elif len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.mean = sum(data) / len(data)
            variance = \
                sum([(x - self.mean) ** 2 for x in data]) / (len(data))
            self.stddev = variance ** .5

    def z_score(self, x):
        """Calculates the z-score of `x`."""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Calculates the x-value of a z-score."""
        return z * self.stddev + self.mean

    def pdf(self, x):
        """The probability density function."""
        exponent = self.z_score(x) ** 2 / 2
        return 1 / (self.stddev * self.tau ** .5 * self.e ** exponent)

    def erf(self, x):
        """The error function."""
        return (
            2 / self.pi ** .5
            * (
                + x
                - x ** 3 / 3
                + x ** 5 / 10
                - x ** 7 / 42
                + x ** 9 / 216
            )
        )

    def cdf(self, x):
        """The cumulative distribution function."""
        error_x = (x - self.mean) / (self.stddev * 2 ** .5)
        return (1 + self.erf(error_x)) / 2

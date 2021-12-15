#!/usr/bin/env python3
"""Defines `Normal`."""


class Normal:
    """A normal distribution."""

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

#!/usr/bin/env python3
"""Defines `Binomial`."""


class Binomial:
    """A binomial distribution."""

    e = 2.7182818285
    pi = 3.1415926536
    tau = 2 * pi

    def __init__(self, data=None, n=1, p=0.5):
        """Initializes a binomial distribution."""
        if data is None:
            if n <= 0:
                raise ValueError('n must be a positive value')
            if p <= 0 or p >= 1:
                raise ValueError('p must be greater than 0 and less than 1')
            self.n = int(n)
            self.p = float(p)
        else:
            if type(data) is not list:
                raise TypeError('data must be a list')
            elif len(data) < 2:
                raise ValueError('data must contain multiple values')
            mean = sum(data) / len(data)
            variance = sum([(x - mean) ** 2 for x in data]) / len(data)
            p = 1 - (variance / mean)
            self.n = round(mean / p)
            self.p = mean / self.n

    def pmf(self, k):
        """The probability mass function."""
        k = int(k)
        if k < 0 or k > self.n:
            return 0

        n_choose_k = self.factorial(self.n) \
            / (self.factorial(k) * self.factorial(self.n - k))

        return n_choose_k * self.p ** k * (1 - self.p) ** (self.n - k)

    def factorial(self, n):
        """The factorial function."""
        if n == 1 or n == 0:
            return 1
        else:
            return n * self.factorial(n-1)

    def cdf(self, k):
        """The cumulative distribution function."""
        k = int(k)
        if k < 0 or k > self.n:
            return 0

        return sum([self.pmf(n) for n in range(k + 1)])

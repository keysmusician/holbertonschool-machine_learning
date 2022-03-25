#!/usr/bin/env python3
"""
You are conducting a study on a revolutionary cancer drug and are looking to
find the probability that a patient who takes this drug will develop severe
side effects. During your trials, n patients take the drug and x patients
develop severe side effects. You can assume that x follows a binomial
distribution.
"""
import numpy as np


def intersection(x, n, P, Pr):
    """
    Calculates the intersection of obtaining the data `x` & `n`, given various
    hypothetical probabilities `P` of developing severe side effects.

    x: The number of patients that develop severe side effects.
    n: The total number of patients observed.
    P: A 1D numpy.ndarray containing the various hypothetical probabilities of
        developing severe side effects.
    Pr: A 1D numpy.ndarray containing the prior beliefs of P.

    Returns: A 1D numpy.ndarray containing the likelihood of obtaining the
        data, `x` and `n`, for each probability in `P`, respectively.
    """
    if type(n) is not int or n < 1:
        raise ValueError('n must be a positive integer')
    if type(x) is not int or x < 0:
        raise ValueError(
            'x must be an integer that is greater than or equal to 0')
    if x > n:
        raise ValueError('x cannot be greater than n')
    if type(P) is not np.ndarray or len(P.shape) != 1:
        raise TypeError('P must be a 1D numpy.ndarray')
    if type(P) is not np.ndarray or Pr.shape != P.shape:
        raise TypeError('Pr must be a numpy.ndarray with the same shape as P')
    for p, pr in zip(P, Pr):
        if p < 0 or p > 1:
            raise ValueError('All values in P must be in the range [0, 1]')
        if pr < 0 or pr > 1:
            raise ValueError('All values in Pr must be in the range [0, 1]')
    if not np.isclose(sum(Pr), 1):
        raise ValueError('Pr must sum to 1')

    factorial = np.math.factorial
    n_choose_x = factorial(n) / (factorial(x) * factorial(n - x))

    likelihoods = np.ones_like(P)
    for i, p in enumerate(P):
        # Binomial distribution:
        likelihoods[i] *= n_choose_x * p ** x * (1 - p) ** (n - x)

    return likelihoods * Pr

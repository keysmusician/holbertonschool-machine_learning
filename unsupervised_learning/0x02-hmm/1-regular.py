#!/usr/bin/env python3
""" Defines `absorbing` """
import numpy as np


def regular(P):
    """
    Determines the steady state probabilities of a regular markov chain.

    P: A is a square 2D numpy.ndarray of shape (n, n) representing the standard
        transition matrix:
        - P[i, j] is the probability of transitioning from state i to state j.
        - n is the number of states in the markov chain.

    Returns: A numpy.ndarray of shape (1, n) containing the steady state
        probabilities, or None on failure.
    """
    if type(P) is not np.ndarray or len(P.shape) != 2:
        return

    power = np.linalg.matrix_power(P, 100)
    if np.all(power > 0):
        return power[[0]]

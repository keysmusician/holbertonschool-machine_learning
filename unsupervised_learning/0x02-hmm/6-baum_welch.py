#!/usr/bin/env python3
""" Defines `baum_welch` """
import numpy as np


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    Performs the Baum-Welch algorithm for a hidden markov model.

    Observations: A numpy.ndarray of shape (T,) that contains the index of the
        observation.
        T: The number of observations.
    Transition: A numpy.ndarray of shape (M, M) that contains the initialized
        transition probabilities.
        M: The number of hidden states.
    Emission: A numpy.ndarray of shape (M, N) that contains the initialized
        emission probabilities.
        N: The number of output states.
    Initial: A numpy.ndarray of shape (M, 1) that contains the initialized
        starting probabilities.
    iterations: The number of times expectation-maximization should be
        performed.

    Returns: The converged (Transition, Emission) on success, or (None, None)
        on failure.
    """
    return (None, None)

#!/usr/bin/env python3
""" Defines `backward` """
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """
    Performs the backward algorithm for a hidden markov model.

    Observation: A numpy.ndarray of shape (T,) that contains the index of the
        observation.
        T: The number of observations.
    Emission: A numpy.ndarray of shape (N, M) containing the emission
        probability of a specific observation given a hidden state.
        Emission[i, j] is the probability of observing `j` given the hidden
        state `i`.
        N: The number of hidden states.
        M: The number of all possible observations.
    Transition: A 2D numpy.ndarray of shape (N, N) containing the transition
        probabilities. Transition[i, j] is the probability of transitioning
        from the hidden state `i` to `j`.
    Initial: A numpy.ndarray of shape (N, 1) containing the probability of
        starting in a particular hidden state.

    Returns: (P, B) on success, or (None, None) on failure.
        P: The likelihood of the observations given the model.
        B: A numpy.ndarray of shape (N, T) containing the backward path
            probabilities. B[i, j] is the probability of generating the future
            observations from hidden state `i` at time `j`.
    """
    return (None, None)

#!/usr/bin/env python3
""" Defines `viterbi` """
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """
    Calculates the most likely sequence of hidden states for a hidden markov
    model.

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

    Returns: (path, P) on success, or (None, None) on failure.
        path: A list of length T containing the most likely sequence of hidden
            states.
        P: The probability of obtaining the path sequence.
    """
    return (None, None)

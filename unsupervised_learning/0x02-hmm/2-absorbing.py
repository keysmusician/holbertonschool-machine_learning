#!/usr/bin/env python3
""" Defines `absorbing` """
import numpy as np


def absorbing(P):
    """
    Determines if a markov chain is absorbing.

    P: A is a square 2D numpy.ndarray of shape (n, n) representing the standard
        transition matrix:
        - P[i, j] is the probability of transitioning from state i to state j.
        - n is the number of states in the markov chain.

    Returns: True if `P` is absorbing, otherwise False.
    """
    if (
            len(P.shape) != 2 or
            P.shape[0] != P.shape[1] or
            np.any(np.sum(P, axis=1)) > 1
            ):
        return False

    partition_index = 0
    while P[partition_index][partition_index] == 1:
        partition_index += 1
        if partition_index == len(P):
            return True

    if partition_index == 0:
        return False

    R = P[partition_index:, :partition_index]
    Q = P[partition_index:, partition_index:]
    QI = np.eye(*Q.shape)

    try:
        F = np.linalg.inv(QI - Q)
    except np.linalg.LinAlgError:
        return False

    if np.any(F * R == 0):
        return True

    return False

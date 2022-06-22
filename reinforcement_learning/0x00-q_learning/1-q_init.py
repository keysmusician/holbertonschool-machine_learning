#!/usr/bin/env python3
""" Defines `q_init`. """
import numpy as np


def q_init(env):
    """
    Initializes a Q-table for an environment.

    env: The gym environment instance.
    Returns: The Q-table as a `numpy.ndarray` of zeros.
    """
    return np.zeros((env.observation_space.n, env.action_space.n))

#!/usr/bin/env python3
""" Defines `q_init`. """
import numpy as np


def epsilon_greedy(Q_table, state, epsilon):
    """
    Uses epsilon-greedy to determine the next action.

    Q_table: A `numpy.ndarray` representing the q-table.
    state: The current state.
    epsilon: The exploration-exploitation threshold.

    Returns: The next action index.
    """
    # If exploring, you should pick the next action with numpy.random.randint
    # from all possible actions
    if np.random.uniform(0, 1) > epsilon:
        # Exploit
        return np.argmax(Q_table[state])
    else:
        # Explore
        # Q_table.shape[1] is the number of possible actions
        return np.random.randint(Q_table.shape[1])

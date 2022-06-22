#!/usr/bin/env python3
""" Defines `play`. """
import gym
import numpy as np


def play(env, Q_table, max_steps=100):
    """
    Plays an episode of Frozen Lake using a trained agent.

    env: The FrozenLakeEnv instance.
    Q_table: A `numpy.ndarray` representing the Q-table.
    max_steps: The maximum number of steps in the episode.

    Returns: The total rewards for the episode.
    """
    # Each state of the board should be displayed via the console
    # You should always exploit the Q-table
    assert isinstance(env, gym.Env)

    state = env.reset()
    total_reward = 0

    for _step in range(max_steps):
        print(env.render(mode='ansi'))

        action = np.argmax(Q_table[state])

        state, reward, done, _info = env.step(action)

        total_reward += reward

        if done:
            print(env.render(mode='ansi'))
            break

    return total_reward

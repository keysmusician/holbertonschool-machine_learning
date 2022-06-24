#!/usr/bin/env python3
""" Defines `train`. """
import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(
    env, Q_table, episodes=5000, max_steps=100, learning_rate=0.1,
    discount_rate=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05
    ):
    """
    Performs Q-learning.

    env: The FrozenLakeEnv instance.
    Q_table: A `numpy.ndarray` containing the Q-table.
    episodes: The total number of episodes to train over.
    max_steps: The maximum number of steps per episode.
    learning_rate: The learning rate.
    discount_rate: The discount rate.
    epsilon: The initial exploration-exploitation threshold for epsilon-greedy.
    min_epsilon: The minimum value that epsilon should decay to.
    epsilon_decay: The decay rate for updating epsilon between episodes.

    Returns: Q_table, total_rewards:
        Q_table: The updated Q-table.
        episode_rewards: A list containing the rewards per episode.
    """
    all_episodes_returns = []

    for episode in range(episodes):
        state = env.reset()

        episode_return = 0

        for _step in range(max_steps):
            action = epsilon_greedy(Q_table, state, epsilon)

            next_state, reward, done, _info = env.step(action)

            # If the game ends due to falling in a hole...
            if done and reward == 0:
                # Update the reward to -1
                reward = -1

            episode_return += reward

            Q_table[state, action] = (
                (1 - learning_rate) * Q_table[state, action] +
                learning_rate * (
                    reward + discount_rate * np.max(Q_table[next_state])
                )
            )

            state = next_state

            if done:
                break

        epsilon = min_epsilon + (epsilon - min_epsilon) * np.exp(
            -epsilon_decay * episode)

        all_episodes_returns.append(episode_return)

    return Q_table, all_episodes_returns

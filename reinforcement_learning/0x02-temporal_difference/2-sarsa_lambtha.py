#!/usr/bin/env python3
""" Defines `sarsa_lambtha`. """
import numpy as np


def sarsa_lambtha(env, Q, λ, episodes=5000, max_steps=100, α=0.1, γ=0.99,
        epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs SARSA(λ).

    env: The OpenAI environment instance.
    Q: A `numpy.ndarray` of shape (s,a) containing the Q table.
        s: The number of states in the environment.
        a: The number of actions in the environment.
    λ: The eligibility trace decay rate (aka trace-decay parameter).
    episodes: The total number of episodes to train over.
    max_steps: The maximum number of steps per episode.
    alpha: The learning rate.
    gamma: The discount rate.
    epsilon: The initial threshold for epsilon greedy.
    min_epsilon: The minimum value that epsilon should decay to.
    epsilon_decay: The decay rate for updating epsilon between episodes.

    Returns: The updated Q table.
    """
    eligibility_trace = np.zeros(Q.shape)

    for episode in range(episodes):
        state = env.reset()
        action = policy(state, Q, epsilon)

        for step in range(max_steps):
            next_state, reward, done, _ = env.step(action)
            next_action = policy(next_state, Q, epsilon)

            eligibility_trace *= γ * λ
            eligibility_trace[state, action] += 1

            TD_error = reward + γ * Q[next_state, next_action] -\
                Q[state, action]

            Q += α * TD_error * eligibility_trace

            if done is True:
                break

            state = next_state
            action = next_action

        epsilon = (min_epsilon +
                   (epsilon - min_epsilon) * np.exp(-epsilon_decay * episode))

    return Q


def policy(state, Q, epsilon):
    """
    Epsilon greedy policy.

    state: The state of the environment.
    Q: The Q table.
    epsilon: The exploration/exploitation trade-off threshold.
    """
    p = np.random.uniform()

    if p > epsilon:
        action = np.argmax(Q[state])
    else:
        action = np.random.randint(Q.shape[1])

    return action

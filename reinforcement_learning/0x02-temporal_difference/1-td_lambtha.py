#!/usr/bin/env python3
""" Defines `td_lambtha`. """
import numpy as np


def td_lambtha(
        environment, V, policy, λ, episodes=5000, max_steps=100, α=0.1,
        γ=0.99):
    """
    Performs the TD(λ) algorithm.

    env: The OpenAI environment instance.
    V: A `numpy.ndarray` of shape (s,) containing the value estimate.
        s: The number of states in the environment.
    policy: A function that takes in a state and returns the next action to
        take.
    λ: The eligibility trace decay rate (aka trace-decay parameter).
    episodes: The total number of episodes to train over.
    max_steps: The maximum number of steps per episode.
    α: The learning rate.
    γ: The discount rate.

    Returns: The updated value estimate.
    """
    eligibility_trace = np.zeros(environment.observation_space.n)

    for _ in range(episodes):
        state = environment.reset()

        # Execute an episode under the given policy
        for _ in range(max_steps):
            next_state, reward, done, _ = environment.step(policy(state))

            eligibility_trace *= γ * λ

            eligibility_trace[state] += 1

            TD_error = reward + γ * V[next_state] - V[state]

            V += α * TD_error * eligibility_trace

            if done:
                break

            state = next_state

    return V


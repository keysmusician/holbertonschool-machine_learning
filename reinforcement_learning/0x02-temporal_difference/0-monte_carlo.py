#!/usr/bin/env python3
"""
Defines `monte_carlo`.

Follows the algorithm shown in:

Reinforcement Learning: An Introduction (Second Edition)
By Richard S. Sutton and Andrew G. Barto
Page 92

http://incompleteideas.net/book/RLbook2020.pdf#page=114
"""
from collections import Counter


def monte_carlo(
        environment, V, policy, episodes=5000, max_steps=100, α=0.1, γ=0.99):
    """
    Performs the first-visit Monte Carlo method of estimating the value
    function `V`.

    environment: The OpenAI environment instance.
    V: A `numpy.ndarray` of shape (s,) containing the value estimate.
        s: The number of states in the environment.
    policy: A function that takes in a state and returns the next action to
        take.
    episodes: The total number of episodes to train over.
    max_steps: The maximum number of steps per episode.
    α: The learning rate.
    γ: The discount rate.

    Returns: The updated value function estimate.
    """
    # The returns for each encountered state across all episodes
    states_returns = {
        state: [] for state in range(environment.observation_space.n)}

    for _ in range(episodes):
        # The states and rewards encountered (in a single episode)
        history = []

        state = environment.reset()

        history.append((state, 0))

        # Execute an episode under the given policy
        for _ in range(max_steps):
            state, reward, done, _ = environment.step(policy(state))

            # Record all the states and rewards encountered
            history.append((state, reward))

            if done:
                break

        # Cumulative discounted reward aka "return"
        Return = 0

        # Count the number of times each state appeared in this episode
        state_visitation_counts = Counter(state for state, _ in history)

        # Iterate over history backwards (to easily compute the return)
        for state, reward in reversed(history):
            # The return from the current state onwards
            Return = γ * Return + reward

            # If this is the first visit to the state
            if state_visitation_counts[state] == 1:
                # The average return across episodes from the state
                state_returns = states_returns[state]

                state_returns.append(Return)

                # The updated state value is the average return across episodes
                # for that state
                V[state] = α * sum(state_returns) / len(state_returns)

            state_visitation_counts[state] -= 1

    return V

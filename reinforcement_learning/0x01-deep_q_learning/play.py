#!/usr/bin/env python3
""" Loads a Deep Q-Network (DQN) to play Atari's "Breakout." """
from train import agent, game_environment


agent.load_weights('policy.h5')

episodes = 3

max_episode_steps = 10_000

for _ in range(episodes):
    state = game_environment.reset()

    for _ in range(max_episode_steps):
        action = agent.forward(state)
        state, _, done, _ = game_environment.step(action)

        if done:
            break

#!/usr/bin/env python3
""" Trains a Deep Q-Network (DQN) to play Atari's "Breakout." """
import gym
from keras import layers, models, optimizers
from rl import agents, memory
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

total_training_steps = 10_000

memory_limit = 1_000_000

game_environment = gym.make(
    "ALE/Breakout-v5",
    disable_env_checker=True,
    obs_type='grayscale',
    render_mode='human',
)

# The number of actions available to take in the game
action_count = game_environment.action_space.n
# The resolution of the game screen in pixels
game_x_resolution, game_y_resolution = game_environment.observation_space.shape
# The number of color channels (only used with "rbg" observation type)
# color_channels = 3

input_shape = (action_count, game_x_resolution, game_y_resolution)

model = models.Sequential([
    layers.Input(shape=input_shape),
    layers.Conv2D(32, 8, strides=4, activation="relu",
        data_format='channels_first'), # Channels are first when grayscale
    layers.Conv2D(64, 4, strides=2, activation="relu",
        data_format='channels_first'),
    layers.Conv2D(64, 3, strides=1, activation="relu",
        data_format='channels_first'),
    layers.Flatten(),
    layers.Dense(512, activation="relu"),
    layers.Dense(action_count, activation="linear"),
])

agent = agents.DQNAgent(
    model=model,
    nb_actions=action_count,
    memory=memory.SequentialMemory(memory_limit, window_length=action_count)
)

agent.compile(optimizers.Adam())

if __name__ == '__main__':

    agent.fit(game_environment, total_training_steps, verbose=2)

    agent.save_weights('policy.h5')

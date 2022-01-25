#!/usr/bin/env python3
"""Defines `save_config` and `load_config`."""
import tensorflow.keras as K


def save_config(network, filename):
    """Saves a model’s configuration in JSON format.

    network: The model whose configuration should be saved.
    filename: The path of the file that the configuration should be saved to.
    """
    json_network = network.to_json()
    with open(filename, 'x') as json_file:
        json_file.write(json_network)


def load_config(filename):
    """
    Loads a model with a specific configuration.

    filename: The path of the file containing the model's configuration in JSON
        format.
    Returns: The loaded model.
    """
    with open(filename, 'r') as json_file:
        json_network = json_file.read()
        json_network = K.models.model_from_json(json_network)

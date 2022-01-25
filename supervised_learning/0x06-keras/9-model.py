#!/usr/bin/env python3
"""Defines `save_model` and `load_model`."""
import tensorflow.keras as K


def save_model(network, filename):
    """
    Saves an entire model.

    network: The model to save.
    filename: The path of the file that the model should be saved to.
    """
    network.save(filename)


def load_model(filename):
    """
    Loads an entire model.

    filename: The path of the file that the model should be loaded from.

    Returns: the loaded model.
    """
    return K.models.load_model(filename)

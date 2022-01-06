#!/usr/bin/env python3
"""Defines `forward_prop`."""
import tensorflow.compat.v1 as tf


create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Creates the forward propagation graph for the neural network.

    Args:
        x: A placeholder for the input data.
        layer_sizes: A list containing the number of nodes in each layer of the
            network
        activations: A list containing the activation functions for each layer
            of the network

    Returns: A tensor for the prediction of the network.
    """
    input = x
    for layer_size, activation_function in zip(layer_sizes, activations):
        input = create_layer(input, layer_size, activation_function)

    return input

#!/usr/bin/env python3
"""Defines `build_model`."""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network with the Keras library:

    nx: The number of input features to the network.
    layers: A list containing the number of nodes in each layer of the network.
    activations: A list containing the activation functions used for each layer
        of the network.
    lambtha: is the L2 regularization parameter.
    keep_prob: The probability that a node will be kept for dropout.

    Returns: The Keras model.
    """
    model = K.Sequential()
    input_shape = (nx,)
    for i, nodes in enumerate(layers):
        layer = K.layers.Dense(
            nodes,
            activation=activations[i],
            kernel_regularizer=K.regularizers.L2(lambtha),
            input_shape=input_shape
        )
        input_shape = tuple()
        model.add(layer)
        dropout = K.layers.Dropout(1-keep_prob)
        model.add(dropout)
    return model

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
    regularizer = K.regularizers.L2(lambtha)
    model.add(K.layers.Dense(
            layers[0], activation=activations[0],
            kernel_regularizer=regularizer,
            input_shape=(nx,)))
    for layer in range(1, len(layers)):
        model.add(K.layers.Dropout(1 - keep_prob))
        model.add(K.layers.Dense(layers[layer], activation=activations[layer],
                                 kernel_regularizer=regularizer))

    return model

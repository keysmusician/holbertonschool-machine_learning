#!/usr/bin/env python3
"""Defines `optimize_model`."""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """
    Sets up Adam optimization for a Keras model with categorical crossentropy
    loss and accuracy metrics.

    network: The model to optimize.
    alpha: The learning rate.
    beta1: The first Adam optimization parameter.
    beta2: The second Adam optimization parameter.
    """
    network.compile(
        optimizer=K.optimizers.Adam(alpha, beta1, beta2),
        loss=K.losses.CategoricalCrossentropy(),
        metrics=[K.metrics.Accuracy()],
    )

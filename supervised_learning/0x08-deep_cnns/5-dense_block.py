#!/usr/bin/env python3
""" Defines `dense_block` """
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    Builds a dense block as described in Densely Connected Convolutional
    Networks.

    X: The output from the previous layer.
    nb_filters: An integer representing the number of filters in X.
    growth_rate: The growth rate for the dense block.
    layers: The number of layers in the dense block.

    Returns: A tuple of:
        - The concatenated output of each layer within the Dense Block, and
        - The number of filters within the concatenated outputs, respectively.
    """

    l_prev = X
    for _ in range(layers):
        bn = K.layers.BatchNormalization()(l_prev)
        A = K.layers.Activation('relu')(bn)
        conv2d = K.layers.Conv2D(
            filters=(4 * growth_rate),
            kernel_size=1,
            padding='same',
            kernel_initializer='he_normal')(A)
        bn2 = K.layers.BatchNormalization()(conv2d)
        A2 = K.layers.Activation('relu')(bn2)
        conv2d = K.layers.Conv2D(
            filters=(growth_rate),
            kernel_size=3,
            padding='same',
            kernel_initializer='he_normal')(A2)
        l_prev = K.layers.Concatenate()([conv2d, l_prev])

    return (l_prev, nb_filters + growth_rate * layers)

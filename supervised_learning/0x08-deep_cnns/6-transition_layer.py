#!/usr/bin/env python3
""" Defines `transition_layer` """
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """
    Builds a transition layer as described in Densely Connected Convolutional
    Networks.

    X: The output from the previous layer.
    nb_filters: An integer representing the number of filters in X.
    compression: The compression factor for the transition layer.

    Returns: A tuple of:
        1) The output of the transition layer, and
        2) the number of filters within the output, respectively.
    """
    filters = int(compression * nb_filters)

    bn = K.layers.BatchNormalization()(X)
    relu = K.layers.ReLU()(bn)
    conv2d = K.layers.Conv2D(
        filters=filters,
        kernel_size=1,
        kernel_initializer='he_normal',
        padding='same'
    )(relu)
    avgpool = K.layers.AveragePooling2D(padding='same')(conv2d,)

    return (avgpool, filters)

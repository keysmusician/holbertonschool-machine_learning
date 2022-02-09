#!/usr/bin/env python3
""" Defines `identity_block` """
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """
    Builds an identity block as described in Deep Residual Learning for Image
    Recognition (2015):

    A_prev: The output from the previous layer.
    filters: A tuple or list containing F11, F3, F12, respectively:
        - F11 is the number of filters in the first 1x1 convolution
        - F3 is the number of filters in the 3x3 convolution
        - F12 is the number of filters in the second 1x1 convolution

    Returns: The activated output of the identity block.
    """
    F11, F3, F12 = filters

    conv_1a = K.layers.Conv2D(
        filters=F11,
        kernel_size=1,
        strides=1,
        padding='same',
        kernel_initializer='he_normal'
    )(A_prev)
    norm_1a = K.layers.BatchNormalization()(conv_1a)
    relu_1a = K.layers.Activation('relu')(norm_1a)

    conv_3 = K.layers.Conv2D(
        filters=F3,
        kernel_size=3,
        strides=1,
        padding='same',
        kernel_initializer='he_normal'
    )(relu_1a)
    norm_3 = K.layers.BatchNormalization()(conv_3)
    relu_3 = K.layers.Activation('relu')(norm_3)

    conv_1b = K.layers.Conv2D(
        filters=F12,
        kernel_size=1,
        strides=1,
        padding='same',
        kernel_initializer='he_normal'
    )(relu_3)
    norm_1b = K.layers.BatchNormalization()(conv_1b)

    add = K.layers.Add()([norm_1b, A_prev])
    return K.layers.Activation('relu')(add)

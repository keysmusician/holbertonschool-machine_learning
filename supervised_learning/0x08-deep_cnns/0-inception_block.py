#!/usr/bin/env python3
""" Defines `inception_block` """
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """
    Builds an inception block as described in Going Deeper with Convolutions
    (2014).

    A_prev: The output from the previous layer
    filters: A tuple or list containing F1, F3R, F3,F5R, F5, FPP, respectively:
        - F1 is the number of filters in the 1x1 convolution
        - F3R is the number of filters in the 1x1 convolution before the 3x3
            convolution
        - F3 is the number of filters in the 3x3 convolution
        - F5R is the number of filters in the 1x1 convolution before the 5x5
            convolution
        - F5 is the number of filters in the 5x5 convolution
        - FPP is the number of filters in the 1x1 convolution after the max
            pooling

    Returns: The concatenated output of the inception block.
    """
    F1, F3R, F3, F5R, F5, FPP = filters

    # Branch 1
    conv2d_1 = K.layers.Conv2D(
        filters=F1,
        kernel_size=1,
        activation='relu',
        padding='same'
    )(A_prev)

    # Branch 2
    conv2d_2 = K.layers.Conv2D(
        filters=F3R,
        kernel_size=1,
        activation='relu',
        padding='same'
    )(A_prev)

    conv2d_3 = K.layers.Conv2D(
        filters=F3,
        kernel_size=3,
        activation='relu',
        padding='same'
    )(conv2d_2)

    # Branch 3
    conv2d_4 = K.layers.Conv2D(
        filters=F5R,
        kernel_size=1,
        activation='relu',
        padding='same'
    )(A_prev)

    conv2d_5 = K.layers.Conv2D(
        filters=F5,
        kernel_size=5,
        activation='relu',
        padding='same'
    )(conv2d_4)

    # Branch 4
    maxpool = K.layers.MaxPool2D(
        pool_size=3,
        strides=1,
        padding='same',
    )(A_prev)

    conv2d_6 = K.layers.Conv2D(
        filters=FPP,
        kernel_size=1,
        activation='relu',
        padding='same'
    )(maxpool)

    # Merge branches
    return K.layers.Concatenate(3)([conv2d_1, conv2d_3, conv2d_5, conv2d_6])

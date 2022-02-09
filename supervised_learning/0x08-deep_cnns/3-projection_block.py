#!/usr/bin/env python3
""" Defines `projection_block` """
import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """
    Builds a projection block as described in Deep Residual Learning for Image
    Recognition (2015).

    A_prev: The output from the previous layer.
    filters: A tuple or list containing F11, F3, F12, respectively.
        - F1_1 is the number of filters in the first 1x1 convolution
        - F3 is the number of filters in the 3x3 convolution
        - F1_2 is the number of filters in the second 1x1 convolution as well
            as the 1x1 convolution in the shortcut connection
    s: is the stride of the first convolution in both the main path and the
        shortcut connection


    Returns: The activated output of the projection block.
    """
    F1_1, F3, F1_2 = filters

    conv2d_1 = K.layers.Conv2D(
        F1_1, 1, s, padding='same', kernel_initializer='he_normal')(A_prev)
    batchnorm_1 = K.layers.BatchNormalization()(conv2d_1)
    A_1 = K.layers.Activation('relu')(batchnorm_1)
    conv2d_2 = K.layers.Conv2D(
        F3, 3, padding='same', kernel_initializer='he_normal')(A_1)
    batchnorm_2 = K.layers.BatchNormalization()(conv2d_2)
    A_2 = K.layers.Activation('relu')(batchnorm_2)
    conv2d_3 = K.layers.Conv2D(
        F1_2, 1, padding='same', kernel_initializer='he_normal')(A_2)
    batchnorm_3 = K.layers.BatchNormalization()(conv2d_3)

    conv2d_shortcut = K.layers.Conv2D(
        F1_2, 1, s, padding='same', kernel_initializer='he_normal')(A_prev)
    batchnorm_shortcut = K.layers.BatchNormalization()(conv2d_shortcut)

    add = K.layers.Add()([batchnorm_3, batchnorm_shortcut])

    return K.layers.Activation('relu')(add)

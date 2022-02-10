#!/usr/bin/env python3
""" Defines `inception_network` """
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    Builds the inception network as described in Going Deeper with Convolutions
    (2014).

    Returns: A Keras Model of the Inception network.
    """
    input_shape = (224, 224, 3)
    X = K.layers.Input(input_shape)

    conv2d_1 = K.layers.Conv2D(
        filters=64,
        kernel_size=7,
        activation='relu',
        strides=2,
        padding='same'
    )(X)

    maxpool_1 = K.layers.MaxPool2D(
        pool_size=3,
        strides=2,
        padding='same'
    )(conv2d_1)

    conv2d_2 = K.layers.Conv2D(
        filters=192,
        kernel_size=3,
        activation='relu',
        strides=1,
        padding='same',
    )(maxpool_1)

    maxpool_2 = K.layers.MaxPool2D(
        pool_size=3,
        strides=2,
        padding='same'
    )(conv2d_2)

    incep_1 = inception_block(maxpool_2, [64, 96, 128, 16, 32, 32])
    incep_2 = inception_block(incep_1, [128, 128, 192, 32, 96, 64])

    maxpool_3 = K.layers.MaxPool2D(
        pool_size=3,
        strides=2,
        padding='same'
    )(incep_2)

    incep_3 = inception_block(maxpool_3, [192, 96, 208, 16, 48, 64])
    incep_4 = inception_block(incep_3, [160, 112, 224, 24, 64, 64])
    incep_5 = inception_block(incep_4, [128, 128, 256, 24, 64, 64])
    incep_6 = inception_block(incep_5, [112, 144, 288, 32, 64, 64])
    incep_7 = inception_block(incep_6, [256, 160, 320, 32, 128, 128])

    maxpool_4 = K.layers.MaxPool2D(
        pool_size=3,
        strides=2,
        padding='same'
    )(incep_7)

    incep_8 = inception_block(maxpool_4, [256, 160, 320, 32, 128, 128])
    incep_9 = inception_block(incep_8, [384, 192, 384, 48, 128, 128])

    avgpool = K.layers.AveragePooling2D(
        pool_size=7,
        strides=1,
        padding='valid'
    )(incep_9)

    dropout = K.layers.Dropout(rate=0.4)(avgpool)
    Y = K.layers.Dense(1000, activation='softmax')(dropout)

    return K.Model(X, Y)

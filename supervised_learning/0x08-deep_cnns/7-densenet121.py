#!/usr/bin/env python3
""" Defines `densenet121` """
import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    Builds the DenseNet-121 architecture as described in Densely Connected
    Convolutional Networks.

    growth_rate: The growth rate.
    compression: The compression factor.

    Returns: A Keras Model of the DenseNet-121 architecture.
    """
    input = K.layers.Input(shape=(224, 224, 3))

    norm_0 = K.layers.BatchNormalization()(input)
    relu_0 = K.layers.ReLU()(norm_0)

    conv_0 = K.layers.Conv2D(
        filters=(growth_rate * 2),
        kernel_size=7,
        strides=2,
        padding='same',
        kernel_initializer='he_normal'
    )(relu_0)

    maxpool = K.layers.MaxPooling2D(
        pool_size=3,
        strides=2,
        padding='same'
    )(conv_0)

    block_0, filter_count = dense_block(maxpool, maxpool.shape[-1], growth_rate, 6)
    transition_0, filter_count = transition_layer(block_0, filter_count, compression)
    block_1, filter_count = dense_block(transition_0, filter_count, growth_rate, 12)
    transition_1, filter_count = transition_layer(block_1, filter_count, compression)
    block_2, filter_count = dense_block(transition_1, filter_count, growth_rate, 24)
    transition_2, filter_count = transition_layer(block_2, filter_count, compression)
    block_3, filter_count = dense_block(transition_2, filter_count, growth_rate, 16)

    avgpool = K.layers.AveragePooling2D(pool_size=7)(block_3)

    output = K.layers.Dense(
        units=1000,
        activation='softmax',
        kernel_initializer='he_normal',
    )(avgpool)

    return K.Model(input, output)

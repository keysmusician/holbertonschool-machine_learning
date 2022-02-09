#!/usr/bin/env python3
""" Defines `projection_block` """
import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    Builds the ResNet-50 architecture as described in Deep Residual Learning
    for Image Recognition (2015):

    Returns: A Keras Model of the ResNet-50 architecture.
    """
    init = K.initializers.he_normal()
    X = K.layers.Input(shape=(224, 224, 3))

    conv_0 = K.layers.Conv2D(
        filters=64,
        kernel_size=7,
        strides=2,
        kernel_initializer=init,
        padding='same'
    )(X)
    norm_0 = K.layers.BatchNormalization()(conv_0)
    relu_0 = K.layers.Activation('relu')(norm_0)
    pool_0 = K.layers.MaxPooling2D(
        pool_size=3,
        strides=2,
        padding='same'
    )(relu_0)

    projection_1a = projection_block(pool_0, [64, 64, 256], s=1)
    identity_1a = identity_block(projection_1a, [64, 64, 256])
    identity_1b = identity_block(identity_1a, [64, 64, 256])

    projection_2a = projection_block(identity_1b, [128, 128, 512])
    identity_2a = identity_block(projection_2a, [128, 128, 512])
    identity_2b = identity_block(identity_2a, [128, 128, 512])
    identity_2c = identity_block(identity_2b, [128, 128, 512])

    projection_3a = projection_block(identity_2c, [256, 256, 1024])
    identity_3a = identity_block(projection_3a, [256, 256, 1024])
    identity_3b = identity_block(identity_3a, [256, 256, 1024])
    identity_3c = identity_block(identity_3b, [256, 256, 1024])
    identity_3d = identity_block(identity_3c, [256, 256, 1024])
    identity_3e = identity_block(identity_3d, [256, 256, 1024])

    projection_4a = projection_block(identity_3e, [512, 512, 2048])
    identity_4a = identity_block(projection_4a, [512, 512, 2048])
    identity_4b = identity_block(identity_4a, [512, 512, 2048])

    pool_1 = K.layers.AveragePooling2D(
        pool_size=7,
        strides=1,
    )(identity_4b)

    Y = K.layers.Dense(
        units=1000,
        activation='softmax',
        kernel_initializer=init
    )(pool_1)

    return K.Model(X, Y)

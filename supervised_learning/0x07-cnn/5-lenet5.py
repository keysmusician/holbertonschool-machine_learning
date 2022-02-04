#!/usr/bin/env python3
"""Defines `lenet5` using Keras."""
import tensorflow.keras as K


def lenet5(X):
    """
    Builds a modified version of the LeNet-5 architecture using keras:

    The model consists of the following layers in order:
    Convolutional layer with 6 kernels of shape 5x5 with same padding
    Max pooling layer with kernels of shape 2x2 with 2x2 strides
    Convolutional layer with 16 kernels of shape 5x5 with valid padding
    Max pooling layer with kernels of shape 2x2 with 2x2 strides
    Fully connected layer with 120 nodes
    Fully connected layer with 84 nodes
    Fully connected softmax output layer with 10 nodes

    X: A Keras Input of shape (m, 28, 28, 1) containing the input images for
        the network:
        - m is the number of images

    Returns: a Keras Model compiled to use Adam optimization (with default
        hyperparameters) and accuracy metrics
    """
    conv2d_1 = K.layers.Conv2D(
        filters=6,
        kernel_size=5,
        padding='same',
        activation='relu',
        kernel_initializer='he_normal')(X)

    maxpool_1 = K.layers.MaxPool2D(2, 2)(conv2d_1)

    conv2d_2 = K.layers.Conv2D(
        filters=16,
        kernel_size=5,
        padding='valid',
        activation='relu',
        kernel_initializer='he_normal')(maxpool_1)

    maxpool_2 = K.layers.MaxPool2D(2, 2)(conv2d_2)

    flat = K.layers.Flatten()(maxpool_2)

    dense_1 = K.layers.Dense(
        units=120,
        activation='relu',
        kernel_initializer='he_normal')(flat)

    dense_2 = K.layers.Dense(
        units=84,
        activation='relu',
        kernel_initializer='he_normal')(dense_1)

    Y = K.layers.Dense(
        units=10,
        activation='softmax',
        kernel_initializer='he_normal')(dense_2)

    model = K.Model(X, Y)
    model.compile(
        optimizer='adam',
        loss="categorical_crossentropy",
        metrics=['accuracy'])

    return model

#!/usr/bin/env python3
"""Defines `lenet5` using TensorFlow."""
import tensorflow.compat.v1 as tf


def lenet5(x, y):
    """
    Builds a modified version of the LeNet-5 architecture using tensorflow.

    The model consists of the following layers in order:
    Convolutional layer with 6 filters of shape 5x5 with same padding
    Max pooling layer with kernels of shape 2x2 with 2x2 strides
    Convolutional layer with 16 kernels of shape 5x5 with valid padding
    Max pooling layer with kernels of shape 2x2 with 2x2 strides
    Fully connected layer with 120 nodes
    Fully connected layer with 84 nodes
    Fully connected softmax output layer with 10 nodes

    x: A tf.placeholder of shape (m, 28, 28, 1) containing the input images for
        the network:
        - m is the number of images
    y: A tf.placeholder of shape (m, 10) containing the one-hot labels for the
        network.

    Returns: A tuple of:
        1) A tensor for the softmax activated output,
        2) A training operation that utilizes Adam optimization (with default,
            hyperparameters),
        3) A tensor for the loss of the netowrk,
        4) A tensor for the accuracy of the network.
    """
    init = tf.keras.initializers.VarianceScaling(scale=2.0)
    conv2d_1 = tf.layers.Conv2D(
        filters=6,
        kernel_size=5,
        activation=tf.nn.relu,
        kernel_initializer=init,
        padding="same"
    )(x)

    maxpool_1 = tf.layers.MaxPooling2D(2, 2)(conv2d_1)

    conv2d_2 = tf.layers.Conv2D(
        filters=16,
        kernel_size=5,
        activation=tf.nn.relu,
        kernel_initializer=init,
        padding="valid"
    )(maxpool_1)

    maxpool_2 = tf.layers.MaxPooling2D(2, 2)(conv2d_2)

    flat_1 = tf.layers.Flatten()(maxpool_2)

    dense_1 = tf.layers.Dense(
        units=120,
        activation=tf.nn.relu,
        kernel_initializer=init
    )(flat_1)

    dense_2 = tf.layers.Dense(
        units=84,
        activation=tf.nn.relu,
        kernel_initializer=init
    )(dense_1)

    dense_3 = tf.layers.Dense(
        units=10,
        kernel_initializer=init
    )(dense_2)

    loss = tf.losses.softmax_cross_entropy(y, dense_3)
    train = tf.train.AdamOptimizer().minimize(loss)

    output = tf.nn.softmax(dense_3)
    equality = tf.math.equal(tf.argmax(y, 1), tf.argmax(output, 1))
    accuracy = tf.math.reduce_mean(tf.cast(equality, tf.float32))

    return (output, train, loss, accuracy)

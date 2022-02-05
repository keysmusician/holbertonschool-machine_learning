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

    Returns:
        - a tensor for the softmax activated output
        - a training operation that utilizes Adam optimization (with default
            hyperparameters)
        - a tensor for the loss of the netowrk
        - a tensor for the accuracy of the network
    """
    init = tf.keras.initializers.VarianceScaling(scale=2.0)
    x = tf.layers.Conv2D(
        6,
        5,
        activation=tf.nn.relu,
        kernel_initializer=init,
        padding="same"
    )(x)

    x = tf.layers.MaxPooling2D(2, 2)(x)

    x = tf.layers.Conv2D(
        16,
        5,
        activation=tf.nn.relu,
        kernel_initializer=init,
        padding="valid"
    )(x)

    x = tf.layers.MaxPooling2D(2, 2)(x)

    x = tf.layers.Flatten()(x)

    x = tf.layers.Dense(
        120,
        activation=tf.nn.relu,
        kernel_initializer=init
    )(x)

    x = tf.layers.Dense(
        84,
        activation=tf.nn.relu,
        kernel_initializer=init
    )(x)

    output = tf.layers.Dense(
        10,
        activation=tf.nn.softmax,
        kernel_initializer=init
    )(x)

    loss = tf.losses.softmax_cross_entropy(y, output)
    train = tf.train.AdamOptimizer().minimize(loss)

    equality = tf.math.equal(tf.argmax(y, 1), tf.argmax(output, 1))
    accuracy = tf.math.reduce_mean(tf.cast(equality, tf.float32))

    return (output, train, loss, accuracy)

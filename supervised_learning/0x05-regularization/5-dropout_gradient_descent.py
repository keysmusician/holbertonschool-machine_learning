#!/usr/bin/env python3
"""Defines `dropout_gradient_descent`."""


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights of a neural network with Dropout regularization using
    gradient descent.

    Y: A one-hot numpy.ndarray of shape (classes, m) that contains the correct
        labels for the data where "classes" is the number of classes "m" is the
        number of data points.
    weights: A dictionary of the weights and biases of the neural network.
    cache: A dictionary of the outputs and dropout masks of each layer of the
        neural network.
    alpha: The learning rate.
    keep_prob: The probability that a node will be kept.
    L: The number of layers of the network.
    """

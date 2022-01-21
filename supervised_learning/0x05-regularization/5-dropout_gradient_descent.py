#!/usr/bin/env python3
"""Defines `dropout_gradient_descent`."""
import numpy as np


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
    m = len(Y[0])
    dz2 = cache['A{}'.format(L)] - Y
    for layer in range(L, 0, -1):
        A = cache['A{}'.format(layer - 1)]
        W = weights['W{}'.format(layer)]
        dz1 = (W.T @ dz2) * (1 - (A * A))
        if layer > 1:
            dz1 *= cache['D{}'.format(layer - 1)] / keep_prob
        dw = dz2 @ A.T / m
        db = np.mean(dz2, axis=1, keepdims=True)
        dz2 = dz1
        weights['W{}'.format(layer)] -= alpha * dw
        weights['b{}'.format(layer)] -= alpha * db

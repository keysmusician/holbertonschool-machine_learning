#!/usr/bin/env python3
"""Defines `l2_reg_gradient_descent`."""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases of a neural network using gradient descent
    with L2 regularization:

    Y: A one-hot numpy.ndarray of shape (classes, m) that contains the correct
        labels for the data where "classes" is the number of classes and "m" is
        the number of data points
    weights: A dictionary of the weights and biases of the neural network.
    cache: A dictionary of the outputs of each layer of the neural network.
    alpha: The learning rate.
    lambtha: The L2 regularization parameter.
    L: The number of layers of the network.
    """
    m = len(Y[0])
    dz2 = cache["A" + str(L)] - Y
    for layer in range(L, 0, -1):
        A = cache["A" + str(layer - 1)]
        W = weights["W" + str(layer)]
        dz1 = W.T @ dz2 * (1 - A * A)
        dw = dz2 @ A.T / m
        dw += W * lambtha / m
        db = np.mean(dz2, axis=1, keepdims=True)
        dz2 = dz1
        weights["W" + str(layer)] -= alpha * dw
        weights["b" + str(layer)] -= alpha * db

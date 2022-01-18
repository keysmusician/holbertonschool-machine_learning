#!/usr/bin/env python3
"""Defines `l2_reg_gradient_descent`."""


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

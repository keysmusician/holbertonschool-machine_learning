#!/usr/bin/env python3
"""Defines `l2_reg_cost`."""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the cost of a neural network with L2 regularization.

    cost: The cost of the network without L2 regularization.
    lambtha: The regularization parameter.
    weights: A dictionary of the weights and biases (numpy.ndarrays) of the
        neural network.
    L: The number of layers in the neural network.
    m: The number of data points used.

    Returns: The cost of the network accounting for L2 regularization.
    """
    sum = 0
    for layer in range(1, L + 1):
        w = weights['W{}'.format(layer)]
        sum += np.sum(w ** 2)
    return cost + lambtha * sum / (2 * m)

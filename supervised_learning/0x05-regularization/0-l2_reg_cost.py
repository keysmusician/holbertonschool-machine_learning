#!/usr/bin/env python3
"""Defines `l2_reg_cost`."""


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

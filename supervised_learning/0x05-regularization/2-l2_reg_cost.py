#!/usr/bin/env python3
"""Defines `l2_reg_cost`."""


def l2_reg_cost(cost):
    """
    Calculates the cost of a neural network with L2 regularization.

    cost: A tensor containing the cost of the network without L2
        regularization.

    Returns: A tensor containing the cost of the network accounting for L2
        regularization.
    """

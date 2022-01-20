#!/usr/bin/env python3
"""Defines `l2_reg_cost`."""
import tensorflow.compat.v1 as tf


def l2_reg_cost(cost):
    """
    Calculates the cost of a neural network with L2 regularization.

    cost: A tensor containing the cost of the network without L2
        regularization.

    Returns: A tensor containing the cost of the network accounting for L2
        regularization.
    """
    return cost + tf.losses.get_regularization_losses()

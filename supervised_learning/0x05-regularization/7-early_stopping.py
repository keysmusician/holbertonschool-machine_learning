#!/usr/bin/env python3
"""Defines `early_stopping`."""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Determines if gradient descent should halt early.

    Early stopping occurs when the validation cost of the network has not
    decreased relative to the optimal validation cost by more than the
    threshold over a specific patience count.

    cost: The current validation cost of the neural network.
    opt_cost: The lowest recorded validation cost of the neural network.
    threshold: The threshold used for early stopping.
    patience: The patience count used for early stopping.
    count: The count of how long the threshold has not been met.

    Returns: A boolean of whether the network should be stopped early, followed
        by the updated count.
    """

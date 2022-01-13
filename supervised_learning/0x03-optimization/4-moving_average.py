#!/usr/bin/env python3
"""Defines `moving_average`."""
import numpy as np


def moving_average(data, beta):
    """
    Calculates the weighted moving average of a data set.

    data: The list of data to calculate the moving average of.
    beta: The weight used for the moving average.

    Returns: A list containing the moving averages of data.
    """
    exponentially_weighted_averages = []
    prev_EWA = 0
    for t, datum in enumerate(data, 1):
        EWA = beta * prev_EWA + (1 - beta) * datum
        unbiased_EWA = EWA / (1 - beta ** t)
        exponentially_weighted_averages.append(unbiased_EWA)
        prev_EWA = EWA
    return exponentially_weighted_averages

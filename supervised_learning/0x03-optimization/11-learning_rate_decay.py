#!/usr/bin/env python3
"""Defines `learning_rate_decay`."""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Updates the learning rate using inverse time decay in numpy.

    alpha: The original learning rate.
    decay_rate: The weight used to determine the rate at which alpha will
        decay.
    global_step: The number of passes of gradient descent that have elapsed.
    decay_step: The number of passes of gradient descent that should occur
        before alpha is decayed further.

    Returns: the updated value for alpha.
    """
    return alpha / (1 + decay_rate * (global_step // decay_step))

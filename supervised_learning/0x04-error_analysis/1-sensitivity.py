#!/usr/bin/env python3
"""Defines `sensitivity`."""
import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity for each class in a confusion matrix.

    confusion: A confusion numpy.ndarray of shape (classes, classes) where row
        indices represent the correct labels and column indices represent the
        predicted labels where "classes" is the number of classes.

    Returns: a numpy.ndarray of shape (classes,) containing the sensitivity of
        each class.
    """
    return np.diagonal(confusion) / np.sum(confusion, axis=1)


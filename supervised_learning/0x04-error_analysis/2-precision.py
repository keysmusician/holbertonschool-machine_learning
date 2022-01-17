#!/usr/bin/env python3
"""Defines `precision`."""
import numpy as np


def precision(confusion):
    """
    Calculates the precision for each class in a confusion matrix.

    confusion: A confusion numpy.ndarray of shape (classes, classes) where row
        indices represent the correct labels and column indices represent the
        predicted labels and "classes" is the number of classes.

    Returns: A numpy.ndarray of shape (classes,) containing the precision of
        each class.
    """
    return np.diagonal(confusion) / np.sum(confusion, axis=0)

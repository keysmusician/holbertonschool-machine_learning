#!/usr/bin/env python3
"""Defines `f1_score`."""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    Calculates the F1 score of a confusion matrix.

    confusion: A confusion numpy.ndarray of shape (classes, classes) where row
        indices represent the correct labels and column indices represent the
        predicted labels.

    Returns: A numpy.ndarray of shape (classes,) containing the F1 score of
        each class.
    """
    V_sensitivity = sensitivity(confusion)
    V_precision = precision(confusion)
    return 2 * V_sensitivity * V_precision / (V_sensitivity + V_precision)

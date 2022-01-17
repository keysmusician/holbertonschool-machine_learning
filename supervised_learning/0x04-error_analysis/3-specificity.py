#!/usr/bin/env python3
"""Defines `specificity`."""
import numpy as np


def specificity(confusion):
    """
    Calculates the specificity for each class in a confusion matrix.

    confusion: A confusion numpy.ndarray of shape (classes, classes) where row
        indices represent the correct labels and column indices represent the
        predicted labels and "classes" is the number of classes.

    Returns: A numpy.ndarray of shape (classes,) containing the specificity of
        each class.
    """
    false_positives = np.sum(confusion, axis=0) - np.diagonal(confusion)
    true_negatives = []
    for i in range(len(confusion)):
        mask = np.ones(confusion.shape, dtype=np.bool8)
        mask[i] = False
        mask[:,i] = False
        true_negatives.append(np.sum(confusion[mask]))
    return true_negatives / (true_negatives + false_positives)

#!/usr/bin/env python3
"""Defines `create_confusion_matrix`."""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix.

    labels: A one-hot numpy.ndarray of shape (m, classes) containing the
        correct labels for each data point where "m" is the number of data
        points and "classes" is the number of classes.
    logits: A one-hot numpy.ndarray of shape (m, classes) containing the
        predicted labels.

    Returns: A confusion numpy.ndarray of shape (classes, classes) with row
        indices representing the correct labels and column indices representing
        the predicted labels.
    """
    class_count = labels.shape[1]
    confusion = np.zeros((class_count, class_count))
    label_indexes = np.argmax(labels, axis=1)
    logit_indexes = np.argmax(logits, axis=1)
    for lable_index, logit_index in zip(label_indexes, logit_indexes):
        confusion[lable_index, logit_index] += 1
    return confusion

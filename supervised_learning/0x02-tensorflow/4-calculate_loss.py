#!/usr/bin/env python3
"""Defines `calculate_loss`."""
import tensorflow.compat.v1 as tf


def calculate_loss(y, y_pred):
    """
    Calculates the softmax cross-entropy loss of a prediction.

    Args:
        y: A placeholder for the labels of the input data.
        y_pred: A tensor containing the network's predictions.

    Returns: A tensor containing the loss of the prediction.
    """
    return tf.losses.softmax_cross_entropy(y, y_pred)

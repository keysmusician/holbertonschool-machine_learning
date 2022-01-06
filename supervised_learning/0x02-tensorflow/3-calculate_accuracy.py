#!/usr/bin/env python3
"""Defines `calculate_accuracy`."""
import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):
    """
    Calculates the accuracy of a model.

    Accuracy is the rate of correct predictions.

    Args:
        y: A placeholder for the labels of the input data.
        y_pred: A tensor containing the network's predictions.

    Returns: A tensor containing the decimal accuracy of the prediction.
    """
    return tf.math.reduce_mean(tf.reduce_sum(tf.multiply(y, y_pred), 1))

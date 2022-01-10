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
    y1 = tf.argmax(y, 1)
    yp1 = tf.argmax(y_pred, 1)
    equality = tf.math.equal(yp1, y1)
    acc = tf.math.reduce_mean(tf.cast(equality, tf.float32))
    return acc

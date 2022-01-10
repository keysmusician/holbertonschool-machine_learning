#!/usr/bin/env python3
"""Defines `evaluate`."""
import tensorflow.compat.v1 as tf


def evaluate(X, Y, save_path):
    """
    Evaluates the output of a neural network.

    Args:
        X: A numpy.ndarray containing the input data to evaluate.
        Y: A numpy.ndarray containing the one-hot labels for X.
        save_path: The location to load the model from.

    Returns: the network's prediction, accuracy, and loss, respectively
    """
    with tf.Session() as session:
        saver = tf.train.import_meta_graph(save_path + ".meta")
        saver.restore(session, save_path)

        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        y_pred = tf.get_collection("y_pred")[0]
        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]

        routine = (y_pred, accuracy, loss)
        return session.run(routine, {x: X, y: Y})

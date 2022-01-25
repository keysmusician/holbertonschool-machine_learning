#!/usr/bin/env python3
"""Defines `train_model`."""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent.

    network: The model to train.
    data: A numpy.ndarray of shape (m, nx) containing the input data.
    labels: A one-hot numpy.ndarray of shape (m, classes) containing the labels
        of data.
    batch_size: The size of the batch used for mini-batch gradient descent.
    epochs: The number of passes through data for mini-batch gradient descent.
    validation_data: The data to validate the model with, if not None.
    early_stopping: A boolean that indicates whether early stopping should be
        used.
    patience: The patience used for early stopping.
    verbose: A boolean that determines if output should be printed during
        training.
    shuffle: A boolean that determines whether to shuffle the batches every
        epoch.

    Returns: The History object generated after training the model.
    """
    callback = None
    if early_stopping and validation_data:
        callback = K.callbacks.EarlyStopping(patience=patience)
    return network.fit(
        data,
        labels,
        batch_size=batch_size,
        callbacks=[callback],
        epochs=epochs,
        shuffle=shuffle,
        verbose=verbose,
        validation_data=validation_data
    )

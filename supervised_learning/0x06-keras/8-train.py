#!/usr/bin/env python3
"""Defines `train_model`."""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                save_best=False, filepath=None, verbose=True, shuffle=False):
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
    learning_rate_decay: A boolean that indicates whether learning rate decay
        should be used.
    alpha: The initial learning rate.
    decay_rate: The decay rate.
    save_best: A boolean indicating whether to save the model after each epoch
        if it is the best. A model is considered the best if its validation
        loss is the lowest that the model has obtained.
    filepath: The file path where the model should be saved.
    verbose: A boolean that determines if output should be printed during
        training.
    shuffle: A boolean that determines whether to shuffle the batches every
        epoch.

    Returns: The History object generated after training the model.
    """
    callbacks = []
    if validation_data is not None:
        if early_stopping:
            callbacks.append(K.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience
            ))
        if learning_rate_decay:
            callbacks.append(K.callbacks.LearningRateScheduler(
                lambda x: alpha / (1 + decay_rate * x),
                verbose=1
            ))
        if save_best:
            callbacks.append(K.callbacks.ModelCheckpoint(
                filepath,
                save_best_only=True
            ))

    return network.fit(
        data,
        labels,
        batch_size=batch_size,
        callbacks=callbacks,
        epochs=epochs,
        shuffle=shuffle,
        verbose=verbose,
        validation_data=validation_data
    )

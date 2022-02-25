#!/usr/bin/env python3
"""Trains a convolutional neural network to classify the CIFAR 10 dataset."""
import tensorflow.keras as K


def preprocess_data(X, Y):
    """
    Pre-processes the data for the model below.

    X: A numpy.ndarray of shape (m, 32, 32, 3) containing the CIFAR 10 data:
        - m: the number of data points
    Y: A numpy.ndarray of shape (m,) containing the CIFAR 10 labels for X

    Returns: A tuple of:
        1) The preprocessed X
        2) The preprocessed Y
"""
    X_p, Y_p = X, Y
    return (X_p, Y_p)


if __name__ == '__main__':
    train_ds, test_ds = K.datasets.cifar10.load_data()

    inputs = K.layers.Input(shape=(32, 32, 3))

    lmbda = K.layers.Lambda(
        lambda x : K.backend.resize_images(x, 150//32, 150//32, "channels_last")
    )(inputs)

    base_model = K.applications.DenseNet121(
        weights="imagenet",  # Load weights pre-trained on ImageNet.
        input_shape=(128, 128, 3),
        include_top=False,
    )  # Do not include the ImageNet classifier at the top.

    # Freeze the base_model
    base_model.trainable = False

    x = base_model(lmbda, training=False)
    x = K.layers.Flatten()(x)
    x = K.layers.Dense(32, 'relu')(x)
    outputs = K.layers.Dense(10, 'softmax')(x)
    model = K.Model(inputs, outputs)

    model.summary()

    model.compile(
        optimizer=K.optimizers.Adam(),
        loss=K.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[K.metrics.SparseCategoricalAccuracy()],
    )

    def learning_rate_schedule(epoch, lr):
        if epoch < 7:
            return lr
        else:
            return lr * 0.9

    epochs = 10

    model.fit(
        train_ds[0],
        train_ds[1],
        batch_size=32,
        epochs=epochs,
        validation_data=test_ds,
        callbacks=[K.callbacks.LearningRateScheduler(learning_rate_schedule)]
    )

    model.save('cifar10.h5')

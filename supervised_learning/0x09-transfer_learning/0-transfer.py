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
    # Load the cifar10 dataset
    train_data, test_data = K.datasets.cifar10.load_data()
    data_shape = (32, 32, 3)

    # Input layer
    input_layer = K.layers.Input(shape=data_shape)

    # Create a layer to scale the input
    image_size = (224, 224)
    resize_layer = input_layer#K.layers.Lambda(
    #    lambda x : K.preprocessing.image.smart_resize(x, image_size)
    #)(input_layer)

    # Use ResNet101 as the base model
    ResNet101 = K.applications.ResNet101(
        include_top=False,
        input_shape=data_shape,
        input_tensor=resize_layer
    )

    # Freeze the model so it does not train
    ResNet101.trainable = False
    resnet_base = ResNet101(resize_layer, training=False)
    output_layer = K.layers.Dense(10)(resnet_base)

    # Build new model
    model = K.Model(input_layer, output_layer)

    model.compile(
        optimizer=K.optimizers.Adam(),
        loss=K.losses.CategoricalCrossentropy(from_logits=True),
        metrics=[K.metrics.CategoricalAccuracy()]
    )
    model.fit(train_data, epochs=4)
    model.save('cifar10.h5')

#!/usr/bin/env python3
""" Defines `autoencoder`. """
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Creates an autoencoder:

    input_dims: An integer containing the dimensions of the model input
    filters: A list containing the number of filters for each convolutional
        layer in the encoder, respectively.
    latent_dims: An integer containing the dimensions of the latent space
        representation

    Returns: (encoder, decoder, auto)
        encoder: The encoder model.
        decoder: The decoder model.
        auto: The full autoencoder model.
    """
    # encoder
    input_layer = keras.Input(input_dims)
    previous_layer = input_layer
    for filter_count in filters:
        conv2d = keras.layers.Conv2D(
            filters=filter_count,
            kernel_size=(3, 3),
            padding='same',
            activation='relu'
        )(previous_layer)
        previous_layer = keras.layers.MaxPool2D(padding='same')(conv2d)

    Encoder = keras.Model(input_layer, previous_layer, name='Encoder')

    # decoder
    latent_space = keras.Input(latent_dims)
    previous_layer = latent_space
    for i, filter_count in enumerate(reversed(filters)):
        padding = 'valid' if i == len(filters) - 1 else 'same'
        conv2d = keras.layers.Conv2D(
            filters=filter_count,
            kernel_size=(3, 3),
            padding=padding,
            activation='relu'
        )(previous_layer)
        previous_layer = keras.layers.UpSampling2D()(conv2d)

    decoder_layers = keras.layers.Conv2D(
        filters=input_dims[2],
        kernel_size=(3, 3),
        padding='same',
        activation='sigmoid'
    )(previous_layer)
    Decoder = keras.Model(latent_space, decoder_layers, name='Decoder')

    # complete autoencoder
    Autoencoder = keras.Model(input_layer, Decoder(Encoder(input_layer)))
    Autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return (Encoder, Decoder, Autoencoder)

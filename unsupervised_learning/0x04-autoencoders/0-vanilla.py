#!/usr/bin/env python3
""" Defines `autoencoder`. """
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layer_sizes, latent_dims):
    """
    Creates an autoencoder:

    input_dims: An integer containing the dimensions of the model input.
    hidden_layer_sizes: A list containing the number of nodes for each hidden
        layer in the encoder, respectively.
    latent_dims: An integer containing the dimensions of the latent space
        representation.

    Returns: (encoder, decoder, auto)
        encoder: The encoder model.
        decoder: The decoder model.
        auto: The full autoencoder model.
    """
    # encoder
    input_layer = keras.Input((input_dims,))
    previous_layer = input_layer
    for node_count in hidden_layer_sizes:
        previous_layer = keras.layers.Dense(node_count, 'relu')(previous_layer)

    encoder_layers = keras.layers.Dense(latent_dims, 'relu')(previous_layer)
    Encoder = keras.Model(input_layer, encoder_layers)

    # decoder
    latent_space = keras.Input((latent_dims,))
    previous_layer = latent_space
    for node_count in reversed(hidden_layer_sizes):
        previous_layer = keras.layers.Dense(node_count, 'relu')(previous_layer)

    decoder_layers = keras.layers.Dense(input_dims, 'sigmoid')(previous_layer)
    Decoder = keras.Model(latent_space, decoder_layers)

    # complete autoencoder
    Autoencoder = keras.Model(input_layer, Decoder(Encoder(input_layer)))
    Autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return (Encoder, Decoder, Autoencoder)

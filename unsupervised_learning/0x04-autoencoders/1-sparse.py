#!/usr/bin/env python3
""" Defines `autoencoder`. """
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layer_sizes, latent_dims, lambtha):
    """
    Creates a sparse autoencoder.

    input_dims: An integer containing the dimensions of the model input.
    hidden_layers: A list containing the number of nodes for each hidden layer
        in the encoder, respectively.
    latent_dims: An integer containing the dimensions of the latent space
        representation.
    lambtha: The regularization parameter used for L1 regularization on the
        encoded output.

    Returns: (encoder, decoder, auto):
        encoder: The encoder model.
        decoder: The decoder model.
        auto: The sparse autoencoder model.
    """
    # encoder
    input_layer = keras.layers.Input(shape=(input_dims,))
    previous_layer = input_layer
    for node_count in hidden_layer_sizes:
        previous_layer = keras.layers.Dense(node_count, 'relu')(previous_layer)

    encoder_layers = keras.layers.Dense(
        latent_dims,
        'relu',
        activity_regularizer=keras.regularizers.l1(lambtha)
    )(previous_layer)
    Encoder = keras.Model(input_layer, encoder_layers)

    # decoder
    latent_space = keras.layers.Input(shape=(latent_dims,))
    previous_layer = latent_space
    for node_count in reversed(hidden_layer_sizes):
        previous_layer = keras.layers.Dense(node_count, 'relu')(previous_layer)

    decoder_layers = keras.layers.Dense(input_dims, 'sigmoid')(previous_layer)
    Decoder = keras.Model(latent_space, decoder_layers)

    # complete autoencoder
    Autoencoder = keras.Model(input_layer, Decoder(Encoder(input_layer)))
    Autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return (Encoder, Decoder, Autoencoder)

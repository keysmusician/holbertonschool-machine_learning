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

    mean_layer = keras.layers.Dense(latent_dims)(previous_layer)
    log_variance_layer = keras.layers.Dense(latent_dims)(previous_layer)

    def normal_sample(inputs):
        """ Draws samples from a normal distribution. """
        mean, log_stddev = inputs
        std_norm = keras.backend.random_normal(
            shape=(keras.backend.shape(mean_layer)[0], latent_dims),
            mean=0, stddev=1)
        sample = mean + keras.backend.exp(log_stddev / 2) * std_norm
        return sample

    sample_layer = keras.layers.Lambda(normal_sample)(
        [mean_layer, log_variance_layer])
    encoder_outputs = [sample_layer, mean_layer, log_variance_layer]
    Encoder = keras.Model(input_layer, encoder_outputs)

    # decoder
    latent_space = keras.Input((latent_dims,))
    previous_layer = latent_space
    for node_count in reversed(hidden_layer_sizes):
        previous_layer = keras.layers.Dense(node_count, 'relu')(previous_layer)

    decoder_layers = keras.layers.Dense(input_dims, 'sigmoid')(previous_layer)
    Decoder = keras.Model(latent_space, decoder_layers)

    def VAE_loss(inputs, outputs):
        """ Custom loss function including a KL divergence term. """
        reconstruction_loss = keras.losses.binary_crossentropy(inputs, outputs)
        reconstruction_loss *= input_dims
        KL_loss = 1 + log_variance_layer - keras.backend.square(mean_layer) \
            - keras.backend.exp(log_variance_layer)
        KL_loss = keras.backend.sum(KL_loss, axis=-1) * -0.5
        total_loss = keras.backend.mean(reconstruction_loss + KL_loss)
        return total_loss

    # complete autoencoder
    Autoencoder = keras.Model(input_layer, Decoder(Encoder(input_layer)[0]))
    Autoencoder.compile(optimizer='adam', loss=VAE_loss)

    return (Encoder, Decoder, Autoencoder)

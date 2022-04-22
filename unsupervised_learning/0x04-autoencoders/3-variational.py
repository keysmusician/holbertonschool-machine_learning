#!/usr/bin/env python3
""" Defines `autoencoder`. """
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
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
    inputs = keras.Input(shape=(input_dims,))
    h = keras.layers.Dense(hidden_layers[0], activation='relu')(inputs)
    for i in range(1, len(hidden_layers)):
        dims = hidden_layers[i]
        h = keras.layers.Dense(dims, activation='relu')(h)
    z_mean = keras.layers.Dense(latent_dims)(h)
    z_log_sigma = keras.layers.Dense(latent_dims)(h)

    def sampling(args):
        """ samplig f'n for vae """
        z_mean, z_log_sigma = args
        epsilon = keras.backend.random_normal(
            shape=(keras.backend.shape(z_mean)[0], latent_dims),
            mean=0, stddev=1)
        return z_mean + keras.backend.exp(z_log_sigma / 2) * epsilon

    z = keras.layers.Lambda(sampling)([z_mean, z_log_sigma])
    encoder = keras.Model(inputs, [z_mean, z_log_sigma, z])

    dinputs = keras.Input(shape=(latent_dims,))
    dh = keras.layers.Dense(hidden_layers[-1], activation='relu')(dinputs)
    for i in range(len(hidden_layers) - 2, -1, -1):
        dims = hidden_layers[i]
        dh = keras.layers.Dense(dims, activation='relu')(dh)
    decode = keras.layers.Dense(input_dims, activation='sigmoid')(dh)

    decoder = keras.Model(dinputs, decode)
    outputs = decoder(encoder(inputs)[-1])
    auto = keras.Model(inputs, outputs)

    def vae_loss(inputs, outputs):
        """ separate custom loss function """
        r_loss = keras.losses.binary_crossentropy(inputs, outputs)
        r_loss *= input_dims
        kl_loss = 1 + z_log_sigma - keras.backend.square(z_mean) \
            - keras.backend.exp(z_log_sigma)
        kl_loss = keras.backend.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = keras.backend.mean(r_loss + kl_loss)
        return vae_loss

    auto.compile(optimizer='adam', loss=vae_loss)

    return (encoder, decoder, auto)

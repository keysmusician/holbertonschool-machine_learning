#!/usr/bin/env python3
""" Defines `Decoder`. """
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


class Decoder(tf.keras.layers.Layer):
    """ A transformer decoder layer. """

    def __init__(
            self, N, dm, h, hidden, target_vocab, max_seq_len, drop_rate=0.1):
        """ Initializes a Decoder layer. """
        super().__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [
            DecoderBlock(dm, h, hidden, drop_rate) for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        Executes a Decoder layer.

        x: A tensor of shape (batch, target_seq_len, dm) containing the input
            to the decoder.
        encoder_output: A tensor of shape (batch, input_seq_len, dm) containing
            the output of the encoder.
        training: A boolean to determine if the model is training.
        look_ahead_mask: The mask to be applied to the first multi head
            attention layer.
        padding_mask: The mask to be applied to the second multi head attention
            layer.

        Returns: A tensor of shape (batch, target_seq_len, dm) containing the
            decoder output.
        """
        x = self.embedding(x) + self.positional_encoding[:x.shape[1]]
        x = self.dropout(x, training=training)

        for decoder_block in self.blocks:
            x = decoder_block(
                x, encoder_output, training, look_ahead_mask, padding_mask)

        return x

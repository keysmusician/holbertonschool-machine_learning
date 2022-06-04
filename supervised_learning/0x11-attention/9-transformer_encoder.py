#!/usr/bin/env python3
""" Defines `Encoder`. """
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
    """ A transformer encoder layer. """

    def __init__(
            self, N, dm, h, hidden, input_vocab, max_seq_len, drop_rate=0.1):
        """
        Initializes an Encoder layer.

        N: The number of blocks in the encoder.
        dm: The dimensionality of the model.
        h: The number of heads.
        hidden: The number of hidden units in the fully connected layer.
        input_vocab: The size of the input vocabulary.
        max_seq_len: The maximum sequence length possible.
        drop_rate: The dropout rate.
        """
        super().__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [
            EncoderBlock(dm, h, hidden, drop_rate) for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """
        Executes an Encoder layer.

        x: A tensor of shape (batch, input_seq_len) containing the input to
            the encoder.
        training: A boolean to determine if the model is training.
        mask: The mask to be applied for multi head attention.

        Returns: A tensor of shape (batch, input_seq_len, dm) containing the
            encoder output.
        """
        x = self.embedding(x) + self.positional_encoding[:x.shape[1]]
        x = self.dropout(x, training=training)

        for encoder_block in self.blocks:
            x = encoder_block(x, training, mask)

        return x

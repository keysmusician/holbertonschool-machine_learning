#!/usr/bin/env python3
""" Defines `DecoderBlock`. """
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """ An encoder block layer. """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """ Initializes a DecoderBlock layer. """
        super().__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        Executes a DecoderBlock layer.

        x: A tensor of shape (batch, target_seq_len, dm) containing the input
            to the decoder block.
        encoder_output: A tensor of shape (batch, input_seq_len, dm) containing
            the output of the encoder.
        training: A boolean to determine if the model is training.
        look_ahead_mask: The mask to be applied to the first multi head
            attention layer.
        padding_mask: The mask to be applied to the second multi head attention
            layer.

        Returns: A tensor of shape (batch, target_seq_len, dm) containing the
            block's output.
        """
        mha1, _ = self.mha1(x, x, x, look_ahead_mask)
        dropout1 = self.dropout1(mha1, training=training)
        layer_norm1 = self.layernorm1(dropout1 + x)
        mha2, _ = self.mha2(
            layer_norm1, encoder_output, encoder_output, padding_mask)
        dropout2 = self.dropout2(mha2, training=training)
        layer_norm2 = self.layernorm2(dropout2 + layer_norm1)
        dense1 = self.dense_hidden(layer_norm2)
        dense2 = self.dense_output(dense1)
        dropout3 = self.dropout3(dense2, training=training)
        layer_norm3 = self.layernorm3(dropout3 + layer_norm2)

        return layer_norm3

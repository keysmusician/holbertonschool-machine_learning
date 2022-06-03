#!/usr/bin/env python3
""" Defines `EncoderBlock`. """
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """ An encoder block layer. """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Initializes a EncoderBlock layer.

        dm: The number of feature dimensions in the model.
        h: The number of heads in the MultiHeadAttention layer.
        hidden: The number of hidden units in the fully connected layer.
        drop_rate: The dropout rate.
        """
        super().__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """
        Executes the EncoderBlock layer.

        x: A tensor of shape (batch, input_seq_len, dm) containing the input to
            the encoder block.
        training: A boolean to determine if the model is training.
        mask: The mask to be applied for multi head attention.

        Returns: A tensor of shape (batch, input_seq_len, dm) containing the
            block's output.
        """
        mha_scores, _ = self.mha(x, x, x, mask)
        dropout1 = self.dropout1(mha_scores, training=training)
        layer_norm1 = self.layernorm1(dropout1 + x)
        dense1 = self.dense_hidden(layer_norm1)
        dense2 = self.dense_output(dense1)
        dropout2 = self.dropout2(dense2, training=training)
        layer_norm2 = self.layernorm2(dropout2 + layer_norm1)

        return layer_norm2

#!/usr/bin/env python3
""" Defines `Transformer`. """
import tensorflow as tf
import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    Calculates the sinusoidal positional encoding for a transformer.

    max_seq_len: An integer representing the maximum sequence length.
    dm: The model depth.

    Returns: A `numpy.ndarray` of shape (max_seq_len, dm) containing the
        positional encoding vectors.
    """
    yy, xx = np.meshgrid(np.arange(dm), np.arange(max_seq_len))

    return np.sin(xx / 10000 ** (yy // 2 * 2 / dm) + np.pi / 2 * (yy % 2))


def sdp_attention(Q, K, V, mask=None):
    """
    Calculates the scaled dot product attention.

    Q: A tensor with its last two dimensions as (..., seq_len_q, dk) containing
        the query matrix.
        dk: The number of feature dimensions in `K`.
    K: A tensor with its last two dimensions as (..., seq_len_v, dk) containing
        the key matrix.
    V: A tensor with its last two dimensions as (..., seq_len_v, dv) containing
        the value matrix.
        dv: The number of feature dimensions in `V`.
    mask: A tensor that can be broadcast into (..., seq_len_q, seq_len_v)
        containing the optional mask, or defaulted to None.

    The preceding dimensions of Q, K, and V must be the same.

    Returns: (output, weights)
        output: A tensor with its last two dimensions as (..., seq_len_q, dv)
            containing the scaled dot product attention.
        weights: A tensor with its last two dimensions as (..., seq_len_q,
            seq_len_v) containing the attention weights.
    """
    QK = tf.matmul(Q, K, transpose_b=True)
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled = QK / tf.sqrt(dk)
    if mask is not None:
        scaled += mask * -1e9

    weights = tf.nn.softmax(scaled, axis=-1)
    output = tf.matmul(weights, V)

    return output, weights


class MultiHeadAttention(tf.keras.layers.Layer):
    """ A multi-head attention layer. """

    def __init__(self, dm, h):
        """
        Initializes a MultiHeadAttention layer.

        dm: An integer divisible by h representing the dimensionality of the
            model.
        h: An integer representing the number of heads.
        """
        super().__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def call(self, Q, K, V, mask):
        """
        Executes the MultiHeadAttention layer.

        Q: A tensor of shape (batch, seq_len_q, dk) containing the input to
            generate the query matrix.
            dk: The number of feature dimensions in `K`.
        K: A tensor of shape (batch, seq_len_v, dk) containing the input to
            generate the key matrix.
        V: A tensor of shape (batch, seq_len_v, dv) containing the input to
            generate the value matrix.
            dv: The number of feature dimensions in `V`.
        mask: Always None.

        Returns: (output, weights)
            output: A tensor with its last two dimensions as (..., seq_len_q,
                dm) containing the scaled dot product attention.
            weights: A tensor with its last three dimensions as (..., h,
                seq_len_q, seq_len_v) containing the attention weights.
        """
        attention_parameters = [
            self.Wq(Q),
            self.Wk(K),
            self.Wv(V)
        ]
        for i, parameter in enumerate(attention_parameters):
            # Split the feature axis into heads x depth, where depth is a
            # subset/slice of the features
            # Then, swap the heads & tokens axes
            attention_parameters[i] = tf.transpose(
                tf.reshape(
                    parameter, (*parameter.shape[:-1], self.h, self.depth)
                ),
                perm=[0, 2, 1, 3]
            )

        attention_scores, weights = sdp_attention(*attention_parameters, mask)
        # Un-swap the heads & tokens axes
        attention_scores = tf.transpose(attention_scores, perm=[0, 2, 1, 3])
        # And merge the heads back into a single features axis
        attention_scores = tf.reshape(
            attention_scores, (*attention_scores.shape[:-2], self.dm))
        output = self.linear(attention_scores)

        return output, weights


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
        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
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
        layer_norm1 = self.layer_norm1(dropout1 + x)
        dense1 = self.dense_hidden(layer_norm1)
        dense2 = self.dense_output(dense1)
        dropout2 = self.dropout2(dense2, training=training)
        layer_norm2 = self.layer_norm2(dropout2 + layer_norm1)

        return layer_norm2


class DecoderBlock(tf.keras.layers.Layer):
    """ An encoder block layer. """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """ Initializes a DecoderBlock layer. """
        super().__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
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
        layer_norm1 = self.layer_norm1(dropout1 + x)
        mha2, _ = self.mha2(
            layer_norm1, encoder_output, encoder_output, padding_mask)
        dropout2 = self.dropout2(mha2, training=training)
        layer_norm2 = self.layer_norm2(dropout2 + layer_norm1)
        dense1 = self.dense_hidden(layer_norm2)
        dense2 = self.dense_output(dense1)
        dropout3 = self.dropout3(dense2)
        layer_norm3 = self.layer_norm3(dropout3 + layer_norm2)

        return layer_norm3


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


class Transformer(tf.keras.Model):
    """ A Transformer model. """

    def __init__(
            self, N, dm, h, hidden, input_vocab, target_vocab, max_seq_input,
            max_seq_target, drop_rate=0.1):
        """
        Initializes a Transformer model.

        N: The number of blocks in the encoder and decoder.
        dm: The dimensionality of the model.
        h: The number of heads.
        hidden: The number of hidden units in the fully connected layers.
        input_vocab: The size of the input vocabulary.
        target_vocab: The size of the target vocabulary.
        max_seq_input: The maximum sequence length possible for the input.
        max_seq_target: The maximum sequence length possible for the target.
        drop_rate: The dropout rate.
        """
        super().__init__()
        self.encoder = Encoder(
            N, dm, h, hidden, input_vocab, max_seq_input, drop_rate)
        self.decoder = Decoder(
            N, dm, h, hidden, input_vocab, max_seq_target, drop_rate)
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(
            self, inputs, target, training, encoder_mask, look_ahead_mask,
            decoder_mask):
        """
        Executes a Transformer model.

        inputs: A tensor of shape (batch, input_seq_len) containing the inputs.
        target: A tensor of shape (batch, target_seq_len) containing the
            target.
        training: A boolean to determine if the model is training.
        encoder_mask: The padding mask to be applied to the encoder.
        look_ahead_mask: The look ahead mask to be applied to the decoder.
        decoder_mask: The padding mask to be applied to the decoder.

        Returns: A tensor of shape (batch, target_seq_len, target_vocab)
            containing the transformer output.
        """
        encoded = self.encoder(inputs, training, encoder_mask)
        decoded = self.decoder(
            target, encoded, training, look_ahead_mask, decoder_mask)

        return self.linear(decoded)

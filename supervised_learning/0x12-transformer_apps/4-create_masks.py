#!/usr/bin/env python3
""" Defines `create_masks`. """
import tensorflow.compat.v2 as tf


def create_masks(inputs, targets):
    """
    Creates all masks for training/validation.

    inputs: A `tf.Tensor` of shape (batch_size, seq_len_in) that contains the
        input sentence.
    targets: A `tf.Tensor` of shape (batch_size, seq_len_out) that contains the
        target sentence.

    Returns: (encoder_mask, combined_mask, decoder_mask)
        encoder_mask: Is the `tf.Tensor` padding mask of shape (batch_size, 1,
            1, seq_len_in) to be applied in the encoder.
        combined_mask: Is the `tf.Tensor` of shape (batch_size, 1, seq_len_out,
            seq_len_out) used in the 1st attention block in the decoder to pad
            and mask future tokens in the input received by the decoder. It
            takes the maximum between a look ahead mask and the decoder target
            padding mask.
        decoder_mask: Is the `tf.Tensor` padding mask of shape (batch_size, 1,
            1, seq_len_in) used in the 2nd attention block in the decoder.
    """
    encoder_mask = tf.cast(
        tf.math.equal(inputs, 0), tf.float32)[:, tf.newaxis, tf.newaxis, :]

    batch_size, input_len = tf.shape(targets)

    combined_mask = tf.repeat(
        1 - tf.linalg.band_part(
                tf.ones((input_len, input_len)), -1, 0)[tf.newaxis],
        batch_size,
        0
    )[:, tf.newaxis]

    decoder_mask = tf.cast(
        tf.math.equal(inputs, 0), tf.float32)[:, tf.newaxis, tf.newaxis, :]

    return encoder_mask, combined_mask, decoder_mask

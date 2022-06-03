#!/usr/bin/env python3
""" Defines `sdp_attention`. """
import tensorflow as tf


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

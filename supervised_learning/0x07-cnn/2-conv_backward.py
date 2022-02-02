#!/usr/bin/env python3
"""Defines `conv_backward`"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs back propagation over a convolutional layer of a neural network.

    dZ: A numpy.ndarray of shape (m, h_new, w_new, c_new) containing the
        partial derivatives with respect to the unactivated output of the
        convolutional layer:
        - m is the number of examples
        - h_new is the height of the output
        - w_new is the width of the output
        - c_new is the number of channels in the output
    A_prev: A numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing the
        output of the previous layer:
        - h_prev is the height of the previous layer
        - w_prev is the width of the previous layer
        - c_prev is the number of channels in the previous layer
    W: A numpy.ndarray of shape (fh, fw, c_prev, c_new) containing the kernels
        for the convolution:
        - fh is the filter height
        - fw is the filter width
    b: A numpy.ndarray of shape (1, 1, 1, c_new) containing the biases applied
        to the convolution.
    padding: A string that is either same or valid, indicating the type of
        padding used.
    stride: A tuple of (sh, sw) containing the strides for the convolution:
        - sh is the stride for the height
        - sw is the stride for the width

    Returns: The partial derivatives with respect to the previous layer
        (dA_prev), the kernels (dW), and the biases (db), respectively.
    """
    train_exmpl_count, input_height, input_width, input_depth = A_prev.shape
    filter_height, filter_width, input_depth, output_depth = W.shape
    train_exmpl_count, output_height, output_width, output_depth = dZ.shape
    vertical_stride, horizontal_stride = stride

    if padding == 'valid':
        padding_height, padding_width = 0, 0
    elif padding == 'same':
        padding_height = np.ceil(((input_height - 1) * vertical_stride -
                                  input_height + filter_height) / 2,
                                 dtype=np.int16)
        padding_width = np.ceil(((input_width - 1) * horizontal_stride -
                                 input_width + filter_width) / 2,
                                dtype=np.int16)

    dA_prev = np.zeros(A_prev.shape)
    dW = np.zeros(W.shape)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)
    print(db.shape)

    for cross_correlation_y in range(filter_height):
        for cross_correlation_x in range(filter_width):

            height_offset = cross_correlation_y * vertical_stride
            width_offset = cross_correlation_x * horizontal_stride

            window = A_prev[
                :,
                height_offset:height_offset + output_height,
                width_offset:width_offset + output_width,
                :
            ]

            # np.sum(window * dZ, axis=(1, 2), keepdims=True)
            dW[
                :,
                cross_correlation_y,
                cross_correlation_x,
                :
            ] += np.tensordot(window, dZ, axes=(1, 2))

    return dA_prev, dW, db

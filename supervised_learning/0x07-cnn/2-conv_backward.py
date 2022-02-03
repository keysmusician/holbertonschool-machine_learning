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
    input_count, input_height, input_width, input_depth = A_prev.shape
    filter_height, filter_width, input_depth, output_depth = W.shape
    input_count, output_height, output_width, output_depth = dZ.shape
    vertical_stride, horizontal_stride = stride

    if padding == 'valid':
        padding_height, padding_width = 0, 0
    elif padding == 'same':
        padding_height = int(np.ceil(((input_height - 1) * vertical_stride -
                                     input_height + filter_height) / 2))
        padding_width = int(np.ceil(((input_width - 1) * horizontal_stride -
                                    input_width + filter_width) / 2))

    padding_shape = (0, padding_height, padding_width, 0)
    padding_shape = tuple(zip(padding_shape))
    A_pad = np.pad(A_prev, padding_shape)

    dA = np.zeros(A_pad.shape)
    dW = np.zeros(W.shape)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    # Normally I use full words for variable names, but i, h, w, and f do not
    # represent an image, height, width, and filter, but rather *indicies* of
    # them. Hence, I will use a single inital letter to signify an index:
    for i in range(input_count):
        for h in range(output_height):
            for w in range(output_width):
                for f in range(output_depth):

                    height_offset = h * vertical_stride
                    width_offset = w * horizontal_stride

                    filter = W[:, :, :, f]
                    dz = dZ[i, h, w, f]
                    window_X = A_pad[
                        i,
                        height_offset:height_offset + filter_height,
                        width_offset:width_offset + filter_width,
                        :
                    ]
                    # Selecting full height, width and depth for each filter
                    dW[:, :, :, f] += window_X * dz

                    dA[
                        i,
                        height_offset:height_offset + filter_height,
                        width_offset:width_offset + filter_width,
                        :
                    ] += dz * filter

    if padding == 'same':
        # Strip off added padding
        dA = dA[
            :,
            padding_height:-padding_height,
            padding_width:-padding_width,
            :
        ]

    return dA, dW, db

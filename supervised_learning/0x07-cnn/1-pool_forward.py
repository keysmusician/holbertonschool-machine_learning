#!/usr/bin/env python3
"""Defines `pool_forward`"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs forward propagation over a pooling layer of a neural network.

    A_prev: A numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing the
        output of the previous layer:
        - m is the number of examples
        - h_prev is the height of the previous layer
        - w_prev is the width of the previous layer
        - c_prev is the number of channels in the previous layer
    kernel_shape: A tuple of (ph, pw) containing the size of the pools:
        - ph is the pool height
        - pw is the pool width
    stride: A tuple of (sh, sw) containing the strides across the pooling:
        - sh is the vertical stride
        - sw is horizontal stride
    mode: A string containing either max or avg, indicating whether to
        perform maximum or average pooling, respectively.

    Returns: The output of the pooling layer.
    """
    training_example_count, input_height, input_width, input_depth = \
        A_prev.shape
    pool_height, pool_width = kernel_shape
    vertical_stride, horizontal_stride = stride

    pooling_shape = (
        training_example_count,
        (input_height - pool_height) // vertical_stride + 1,
        (input_width - pool_width) // horizontal_stride + 1,
        input_depth
    )

    pooling = np.zeros(pooling_shape)
    for pool_y in range(pooling_shape[1]):
        for pool_x in range(pooling_shape[2]):
            vertical_offset = pool_y * vertical_stride
            horizontal_offset = pool_x * horizontal_stride
            window = A_prev[
                :,
                vertical_offset:vertical_offset + pool_height,
                horizontal_offset:horizontal_offset + pool_width,
                :
            ]

            pooling_function = np.max if mode == 'max' else np.mean

            pooling[
                :,
                pool_y,
                pool_x,
                :
            ] = pooling_function(window, axis=(1, 2))

    return pooling

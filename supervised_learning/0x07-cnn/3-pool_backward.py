#!/usr/bin/env python3
"""Defines `pool_backward`"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs back propagation over a pooling layer of a neural network.

    dA: A numpy.ndarray of shape (m, h_new, w_new, c_new) containing the
        partial derivatives with respect to the output of the pooling layer:
        - m is the number of examples
        - h_new is the height of the output
        - w_new is the width of the output
        - c is the number of channels
    A_prev: A numpy.ndarray of shape (m, h_prev, w_prev, c) containing the
        output of the previous layer:
        - h_prev is the height of the previous layer
        - w_prev is the width of the previous layer
    kernel_shape: A tuple of (kh, kw) containing the size of the kernel for the
        pooling:
        - kh is the kernel height
        - kw is the kernel width
    stride: A tuple of (sh, sw) containing the strides for the pooling:
        - sh is the stride for the height
        - sw is the stride for the width
    mode: A string containing either max or avg, indicating whether to perform
        maximum or average pooling, respectively.

    Returns: The partial derivatives of the pooling layer with respect to the
        previous layer.
    """
    input_count, output_height, output_width, output_depth = dA.shape
    vertical_stride, horizontal_stride = stride
    filter_height, filter_width = kernel_shape
    avg_dA = dA / (filter_height * filter_width)
    dX = np.zeros_like(A_prev)

    for dX_y in range(output_height):
        for dX_x in range(output_width):
            vertical_offset = dX_y * vertical_stride
            horizontal_offset = dX_x * horizontal_stride

            if mode == 'max':
                window = A_prev[
                    :,
                    vertical_offset:vertical_offset + filter_height,
                    horizontal_offset:horizontal_offset + filter_width,
                    :
                ]
                mask = window == np.max(window, axis=(1, 2), keepdims=True)
                dx = mask * np.expand_dims(dA[:, dX_y, dX_x, :], axis=(1, 2))

            if mode == 'avg':
                dx = np.expand_dims(avg_dA[:, dX_y, dX_x, :], axis=(1, 2))

            dX[
                :,
                vertical_offset:vertical_offset + filter_height,
                horizontal_offset:horizontal_offset + filter_width,
                :
            ] += dx

    return dX

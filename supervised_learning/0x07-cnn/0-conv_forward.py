#!/usr/bin/env python3
"""Defines `conv_forward`"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Performs forward propagation over a convolutional layer of a neural
    network.

    A_prev: A numpy.ndarray containing the output of the previous layer of
        shape (m, h_prev, w_prev, c_prev):
        - m is the number of training examples
        - h_prev is the height of the previous layer
        - w_prev is the width of the previous layer
        - c_prev is the number of channels in the previous layer
    W: A numpy.ndarray containing the filters for the convolution of shape
        (fh, fw, c_prev, c_new):
        - fh is the filter height
        - fw is the filter width
        - c_prev is the number of channels in the previous layer
        - c_new is the number of channels in the output
    b: A numpy.ndarray containing the biases applied to the convolution of
        shape (1, 1, 1, c_new).
    activation: An activation function applied to the convolution.
    padding: A string that is either "same" or "valid," indicating the type of
        padding used.
    stride: A tuple of (sh, sw) containing the strides across the convolution:
        - sh is the vertical stride
        - sw is horizontal stride

    Returns: The output of the convolutional layer.
    """
    training_example_count, input_height, input_width, input_depth = \
        A_prev.shape
    # filter_depth must equal input_depth and filter_count equals output_depth
    filter_height, filter_width, filter_depth, filter_count = W.shape
    vertical_stride, horizontal_stride = stride

    if padding == 'valid':
        padding_height = 0
        padding_width = 0
    elif padding == 'same':
        padding_height = np.ceil(((input_height - 1) * vertical_stride -
                                  input_height + filter_height) / 2,
                                 dtype=np.int16)
        padding_width = np.ceil(((input_width - 1) * horizontal_stride -
                                 input_width + filter_width) / 2,
                                dtype=np.int16)

    cross_correlation_shape = (
        training_example_count,
        (input_height + 2 * padding_height - filter_height) // vertical_stride
        + 1,
        (input_width + 2 * padding_width - filter_width) // horizontal_stride
        + 1,
        filter_count
    )
    cross_correlation = np.zeros(cross_correlation_shape)

    for cross_correlation_y in range(cross_correlation_shape[1]):
        for cross_correlation_x in range(cross_correlation_shape[2]):

            height_offset = cross_correlation_y * vertical_stride
            width_offset = cross_correlation_x * horizontal_stride

            # Slice the input (the activation matrix of the previous layer)
            window = A_prev[
                :,
                height_offset:height_offset + filter_height,
                width_offset:width_offset + filter_width,
                :
            ]

            cross_correlation[
                :,
                cross_correlation_y,
                cross_correlation_x,
                :
            ] = np.tensordot(window, W, axes=[(1, 2, 3), (0, 1, 2)])

    return activation(cross_correlation + b)

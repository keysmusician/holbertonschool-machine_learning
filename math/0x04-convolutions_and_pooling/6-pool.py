#!/usr/bin/env python3
"""Defines `pool`"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    performs pooling on images:

    images: A numpy.ndarray with shape (m, h, w, c) containing multiple images.
        m: The number of images.
        h: The height in pixels of the images.
        w: The width in pixels of the images.
        c: The number of channels in the image.
    kernel_shape: Is a tuple of (kh, kw) containing the kernel shape for the
        pooling.
        kh: The height of the kernel.
        kw: The width of the kernel.
    stride: A tuple of (sh, sw).
        sh: The vertical stride across the image.
        sw: The horizontal stride across the image.
    mode: Indicates the type of pooling.
        max: Indicates max pooling.
        avg: Indicates average pooling.

    Returns: A numpy.ndarray containing the pooled images.
    """
    image_count, image_height, image_width, image_channels = images.shape
    kernel_height, kernel_width = kernel_shape
    stride_height, stride_width = stride

    convolved_shape = (
        image_count,
        (image_height - kernel_height) // stride_height + 1,
        (image_width - kernel_width) // stride_width + 1,
        image_channels
    )

    convolved = np.zeros(convolved_shape)
    for top in range(convolved_shape[1]):
        for left in range(convolved_shape[2]):
            y = top * stride_height
            x = left * stride_width
            view = images[:, y:y + kernel_height, x:x + kernel_width, :]
            if mode == 'max':
                convolved[:, top, left, :] = np.max(view, axis=(1, 2))
            elif mode == 'avg':
                convolved[:, top, left, :] = np.mean(view, axis=(1, 2))

    return convolved

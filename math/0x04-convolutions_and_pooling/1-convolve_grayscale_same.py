#!/usr/bin/env python3
"""Defines `convolve_grayscale_same`"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on grayscale images.

    images: A numpy.ndarray with shape (m, h, w) containing multiple grayscale
        images.
        - m is the number of images.
        - h is the height in pixels of the images.
        - w is the width in pixels of the images.
    kernel: A numpy.ndarray with shape (kh, kw) containing the kernel for the
        convolution.
        - kh is the height of the kernel.
        - kw is the width of the kernel.

    Returns: A numpy.ndarray containing the convolved images.
    """
    kernel_height, kernel_width = kernel.shape
    image_count, image_height, image_width = images.shape

    pad_height = (kernel_height - 1) // 2
    pad_width = (kernel_width - 1) // 2
    padding = (0, pad_height, pad_width)
    padding = tuple(zip(padding, padding))
    padded_images = np.pad(images, padding)

    output_shape = (
        image_count,
        image_height + 2 * pad_height - kernel_height + 1,
        image_width + 2 * pad_width - kernel_width + 1,
    )
    output = np.zeros(output_shape)
    for top in range(output_shape[1]):
        for left in range(output_shape[2]):
            view = padded_images[
                :,
                top:top + kernel_height,
                left:left + kernel_width
            ]
            output[:, top, left] = np.sum(view * kernel, axis=(1, 2))

    return output

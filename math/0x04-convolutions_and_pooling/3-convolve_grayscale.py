#!/usr/bin/env python3
"""Defines `convolve_grayscale`"""
import numpy as np


def ceil(a):
    b = a // 1
    if a != b:
        return b + 1
    return a


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on grayscale images with custom padding.

    images: A numpy.ndarray with shape (m, h, w) containing multiple grayscale
        images.
        - m is the number of images.
        - h is the height in pixels of the images.
        - w is the width in pixels of the images.
    kernel: A numpy.ndarray with shape (kh, kw) containing the kernel of the
        convolution.
        - kh is the height of the kernel.
        - kw is the width of the kernel.
    padding: Either a tuple of (ph, pw), 'same', or 'valid'.
        - If 'same', performs a same convolution
        - If 'valid', performs a valid convolution
        - If a tuple:
            - ph is the padding on the height of the image.
            - pw is the padding on the width of the image.
    stride: A tuple of (sh, sw).
        - sh is the stride across the height of the image.
        - sw is the stride across the width of the image.

    Returns: A numpy.ndarray containing the convolved images.
    """
    kernel_height, kernel_width = kernel.shape
    image_count, image_height, image_width = images.shape
    stride_height, stride_width = stride

    if padding == 'same':
        pad_height = ceil((
            stride_height * (image_height - 1) - image_height + kernel_height
            ) / 2)
        pad_width = ceil((
            stride_width * (image_width - 1) - image_width + kernel_width) / 2)

        convolved_shape = images.shape
    else:
        if type(padding) is tuple:
            pad_height, pad_width = padding
        elif padding == 'valid':
            pad_width = pad_height = 0

        convolved_shape = (
            image_count,
            (image_height + 2 * pad_height - kernel_height)
            // stride_height + 1,
            (image_width + 2 * pad_width - kernel_width) // stride_width + 1,
        )

    pad_params = (0, pad_height, pad_width)
    pad_params = tuple(zip(pad_params, pad_params))
    padded_images = np.pad(images, pad_params)

    # Matrix of convolved images, zero-initialized:
    convolved = np.zeros(convolved_shape)
    for top in range(convolved_shape[1]):
        for left in range(convolved_shape[2]):
            y = top * stride_height
            x = left * stride_width
            view = padded_images[:, y:y + kernel_height, x:x + kernel_width]
            convolved[:, top, left] = np.sum(view * kernel, axis=(1, 2))

    return convolved

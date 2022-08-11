#!/usr/bin/env python3
""" Defines `crop_image`. """
import tensorflow as tf


def crop_image(image, size):
    """
    Randomly crops an image to a given size.

    image: A 3D `tf.Tensor` containing the image to modify.
    size: A tuple containing the size of the crop.

    Returns: The cropped image.
    """
    return tf.image.random_crop(image, size)

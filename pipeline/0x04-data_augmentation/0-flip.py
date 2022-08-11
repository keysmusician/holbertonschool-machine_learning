#!/usr/bin/env python3
""" Defines `flip_image`. """
import tensorflow as tf


def flip_image(image):
    """
    Flips an image horizontally.

    image: A 3D `tf.Tensor` containing the image to modify.

    Returns: The flipped image.
    """
    return tf.image.flip_left_right(image)

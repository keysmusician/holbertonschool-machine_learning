#!/usr/bin/env python3
""" Defines `rotate_image`. """
import tensorflow as tf


def rotate_image(image):
    """
    Rotates an image by 90 degrees counter-clockwise.

    image: A 3D `tf.Tensor` containing the image to modify.

    Returns: The rotated image.
    """
    return tf.image.rot90(image)

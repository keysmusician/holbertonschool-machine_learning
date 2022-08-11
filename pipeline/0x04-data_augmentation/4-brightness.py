#!/usr/bin/env python3
""" Defines `change_brightness`. """
import tensorflow as tf


def change_brightness(image, max_delta):
    """
    Randomly changes the brightness of an image.

    image: A 3D `tf.Tensor` containing the image to modify.
    max_delta: The maximum amount the image should be brightened or darkened.

    Returns: The brightness-adjusted image.
    """
    return tf.image.random_brightness(image, max_delta)

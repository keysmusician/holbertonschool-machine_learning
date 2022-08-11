#!/usr/bin/env python3
""" Defines `change_hue`. """
import tensorflow as tf


def change_hue(image, delta):
    """
    Changes the hue of an image.

    image: A 3D `tf.Tensor` containing the image to modify.
    delta: The amount the hue should change.

    Returns: The hue-adjusted image.
    """
    return tf.image.adjust_hue(image, delta)

#!/usr/bin/env python3
""" Defines `shear_image`. """
import tensorflow as tf


def shear_image(image, intensity):
    """
    Randomly shears an image.

    image: A 3D `tf.Tensor` containing the image to modify.
    intensity: The intensity of the shear.

    Returns: The sheared image.
    """
    return tf.keras.preprocessing.image.random_shear(image, intensity, 1, 0, 2)

#!/usr/bin/env python3
""" Defines `pca_color`. """
import tensorflow as tf
import numpy as np


def pca_color(image, alphas):
    """
    Performs PCA color augmentation as described in the AlexNet paper.

    image: A 3D `tf.Tensor` containing the image to modify.
    alphas: A tuple of length 3 containing the amount that each channel should
    change.

    Returns: The augmented image.
    """
    R, G, B, = 0, 1, 2

    image_size = image.shape[0] * image.shape[1]

    alphas = tf.cast(alphas, tf.complex128)

    pixel_covarience = np.cov(
        [
            tf.reshape(image[:, :, R], image_size),
            tf.reshape(image[:, :, G], image_size),
            tf.reshape(image[:, :, B], image_size)
        ]
    )

    eigenvalues, eigenvectors = tf.linalg.eig(pixel_covarience)

    intensity = tf.matmul(
                [eigenvalues],
                tf.transpose([alphas * eigenvectors[0]])
    )

    intensity = tf.cast(tf.math.real(intensity), tf.uint8)

    return image + intensity

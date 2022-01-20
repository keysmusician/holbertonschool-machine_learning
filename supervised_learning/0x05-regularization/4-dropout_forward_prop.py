#!/usr/bin/env python3
"""Defines `dropout_forward_prop`."""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using Dropout.

    All layers except the last use the tanh activation function. The last layer
    uses the softmax activation function.

    X: A numpy.ndarray of shape (nx, m) containing the input data for the
        network where "nx" is the number of input features and "m" is the
        number of data points.
    weights: A dictionary of the weights and biases of the neural network.
    L: The number of layers in the network.
    keep_prob: The probability that a node will be kept.

    Returns: A dictionary containing the outputs of each layer and the dropout
        mask used on each layer.
    """
    # Cache of activations and dropout masks:
    cache = {'A0': X}
    # Previous layer activation:
    A = X
    for layer in range(1, L + 1):
        # Weights:
        W = weights["W{}".format(layer)]
        # Bias:
        b = weights["b{}".format(layer)]
        # Linear transformation:
        Z = W @ A + b
        if layer < L:
            # Activation:
            A = np.tanh(Z)
            # Dropout mask:
            D = np.random.rand(*A.shape) < keep_prob
            D = np.where(D, 1, 0)
            cache['D{}'.format(layer)] = D
            # Apply dropout mask:
            A *= D / keep_prob
        else:
            # No dropout on final layer; softmax activation:
            A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
        cache['A{}'.format(layer)] = A

    return cache

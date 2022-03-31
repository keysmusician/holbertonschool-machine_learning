#!/usr/bin/env python3
""" Defines `expectation`. """
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Calculates the expectation step in the EM algorithm for a GMM.

    X: A numpy.ndarray of shape (n, d) containing the data set.
    pi: A numpy.ndarray of shape (k,) containing the priors for each cluster.
    m: A numpy.ndarray of shape (k, d) containing the centroid means for each
        cluster.
    S: A numpy.ndarray of shape (k, d, d) containing the covariance matrices
        for each cluster.

    Returns: (g, l) on success, or (None, None) on failure:
        g: A numpy.ndarray of shape (k, n) containing the posterior
            probabilities for each data point in each cluster.
        l: The total log likelihood.
    """
    if (
            type(X) is not np.ndarray or
            len(X.shape) != 2 or
            type(pi) is not np.ndarray or
            len(pi.shape) != 1 or
            not np.isclose([np.sum(pi)], [1])[0]
            ):
        return (None, None)
    n, d = X.shape
    k = pi.shape[0]
    if (
            type(m) is not np.ndarray or
            m.shape != (k, d) or
            type(S) is not np.ndarray or
            S.shape != (k, d, d)
            ):
        return (None, None)

    numerator = np.zeros((k, n))
    for cluster_index in range(k):
        centroid = m[cluster_index]
        covariance = S[cluster_index]
        prior = pi[cluster_index]
        numerator[cluster_index] = prior * pdf(X, centroid, covariance)

    denominator = np.sum(numerator, axis=0, keepdims=True)
    responsibilities = numerator / denominator

    return (responsibilities, np.sum(np.log(denominator)))

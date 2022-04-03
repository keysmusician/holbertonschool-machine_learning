#!/usr/bin/env python3
""" Defines `expectation_maximization`. """
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Performs the expectation maximization for a GMM.

    X: A numpy.ndarray of shape (n, d) containing the data set.
    k: A positive integer containing the number of clusters.
    iterations: A positive integer containing the maximum number of iterations
        for the algorithm.
    tol: A non-negative float containing tolerance of the log likelihood, used
        to determine early stopping.
    verbose: A boolean that determines if you should print information about
        the algorithm.

    Returns: (pi, m, S, g, l) on success, or (None, None, None, None, None) on
        failure:
        pi: A numpy.ndarray of shape (k,) containing the priors for each
            cluster.
        m: A numpy.ndarray of shape (k, d) containing the centroid means for
            each cluster.
        S: A numpy.ndarray of shape (k, d, d) containing the covariance
            matrices for each cluster.
        g: A numpy.ndarray of shape (k, n) containing the probabilities for
            each data point in each cluster.
        l: The log likelihood of the model.
    """
    if (
            type(X) is not np.ndarray or
            len(X.shape) != 2 or
            type(k) is not int or
            k < 1 or
            type(iterations) is not int or
            iterations < 1 or
            type(tol) is not float or
            tol < 0 or
            type(verbose) is not bool
            ):
        return (None, ) * 5
    priors, centroids, covariances = initialize(X, k)
    previous_log_likelihood = 0
    responsibilities, log_likelihood = expectation(
        X, priors, centroids, covariances)
    for iteration in range(iterations):
        previous_log_likelihood = log_likelihood
        if verbose and iteration % 10 == 0:
            print('Log Likelihood after {} iterations: {:.5f}'.format(
                iteration, log_likelihood))
        priors, centroids, covariances = maximization(X, responsibilities)
        responsibilities, log_likelihood = expectation(
            X, priors, centroids, covariances)
        if abs(log_likelihood - previous_log_likelihood) <= tol:
            break

    if verbose:
        print('Log Likelihood after {} iterations: {:.5f}'.format(
            iteration + 1, log_likelihood))

    return (priors, centroids, covariances, responsibilities, log_likelihood)

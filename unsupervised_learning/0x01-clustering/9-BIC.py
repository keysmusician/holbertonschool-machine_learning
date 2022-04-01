#!/usr/bin/env python3
""" Defines `BIC`. """
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Finds the best number of clusters for a GMM using the Bayesian Information
    Criterion.

    X: A numpy.ndarray of shape (n, d) containing the data set.
    kmin: A positive integer containing the minimum number of clusters to check
        for (inclusive).
    kmax: A positive integer containing the maximum number of clusters to check
        for (inclusive).
    iterations: A positive integer containing the maximum number of iterations
        for the EM algorithm.
    tol: A non-negative float containing the tolerance for the EM algorithm.
    verbose: A boolean that determines if the EM algorithm should print
        information to the standard output.

    Returns: (best_k, best_result, l, b) on success, or (None, None, None,
        None) on failure:
        best_k: The best value for k based on its BIC.
        best_result: A tuple containing (pi, m, S).
            pi: A numpy.ndarray of shape (k,) containing the cluster priors for
                the best number of clusters.
            m: A numpy.ndarray of shape (k, d) containing the centroid means
                for the best number of clusters.
            S: A numpy.ndarray of shape (k, d, d) containing the covariance
                matrices for the best number of clusters.
        l: A numpy.ndarray of shape (kmax - kmin + 1) containing the log
            likelihood for each cluster size tested.
        b: A numpy.ndarray of shape (kmax - kmin + 1) containing the BIC value
            for each cluster size tested.
    """
    fail = (None, None, None, None)
    if (
            type(X) is not np.ndarray or
            len(X.shape) != 2 or
            type(kmin) is not int or
            kmin < 1
            ):
        return fail
    if kmax is None:
        kmax = X.shape[0]
    if (
            type(kmax) is not int or
            kmax < 1 or
            kmax < kmin + 1 or
            type(iterations) is not int or
            iterations < 1 or
            type(tol) is not float or
            tol < 0 or
            type(verbose) is not bool
            ):
        return fail

    n, d = X.shape
    results = []
    log_likelihoods = []
    BIC = []
    for cluster_count in range(kmin, kmax + 1):
        priors, centroids, covariances, responsibilities, log_likelihood = \
            expectation_maximization(
                X, cluster_count, iterations, tol, verbose)
        results.append((priors, centroids, covariances))
        log_likelihoods.append(log_likelihood)
        BIC.append(-2 * log_likelihood + np.log(n) * cluster_count)

    best_cluster_count = np.argmin(BIC)
    best_parameters = results[best_cluster_count]

    return (best_cluster_count, best_parameters, log_likelihoods, BIC)

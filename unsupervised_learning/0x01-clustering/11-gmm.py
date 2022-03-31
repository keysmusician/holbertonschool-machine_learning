#!/usr/bin/env python3
""" Defines `gmm`. """
import sklearn.mixture


def gmm(X, k):
    """
    Calculates a GMM from a dataset.

    X: A numpy.ndarray of shape (n, d) containing the dataset.
    k: The number of clusters.

    Returns: (pi, m, S, clss, bic):
        pi: A numpy.ndarray of shape (k,) containing the cluster priors.
        m: A numpy.ndarray of shape (k, d) containing the centroid means.
        S: A numpy.ndarray of shape (k, d, d) containing the covariance
            matrices.
        clss: A numpy.ndarray of shape (n,) containing the cluster indices for
            each data point.
        bic: A numpy.ndarray of shape (kmax - kmin + 1) containing the BIC
            value for each cluster size tested.
    """
    GMM = sklearn.mixture.GaussianMixture(k).fit(X)
    return (
        GMM.weights_,
        GMM.means_,
        GMM.covariances_,
        GMM.predict(X),
        GMM.bic(X),
    )

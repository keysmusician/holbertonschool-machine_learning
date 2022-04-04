#!/usr/bin/env python3
""" Defines `agglomerative`. """
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
    Performs agglomerative clustering with Ward linkage on a dataset.

    Displays the dendrogram with each cluster displayed in a different color.

    X: A numpy.ndarray of shape (n, d) containing the dataset.
    dist: The maximum cophenetic distance for all clusters.

    Returns: A numpy.ndarray of shape (n,) containing the cluster indices for
        each data point.
    """
    Z = scipy.cluster.hierarchy.linkage(X, 'ward')
    scipy.cluster.hierarchy.dendrogram(Z, color_threshold=dist)
    fig1, ax1 = plt.subplots()
    return scipy.cluster.hierarchy.fcluster(Z, dist, 'distance')

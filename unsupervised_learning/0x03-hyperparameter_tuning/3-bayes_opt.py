#!/usr/bin/env python3
""" Defines `BayesianOptimization`."""
import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Performs Bayesian optimization on a noiseless 1D Gaussian process."""

    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1, sigma_f=1,
                 xsi=0.01, minimize=True):
        """
        Initializes a Bayesian optimizer.

        f: The black-box function to be optimized.
        X_init: A numpy.ndarray of shape (t, 1) representing the inputs
             already sampled with the black-box function.
             t: The number of initial samples.
        Y_init: A numpy.ndarray of shape (t, 1) representing the outputs of
             the black-box function for each input in X_init.
        bounds: A tuple of (min, max) representing the bounds of the space in
             which to look for the optimal point.
        ac_samples: The number of samples that should be analyzed during
             acquisition.
        l: The length parameter for the kernel.
        sigma_f: The standard deviation given to the output of the black-box
             function.
        xsi: The exploration-exploitation factor for acquisition.
        minimize: A bool determining whether optimization should be performed
             for minimization (True) or maximization (False).
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(*bounds, ac_samples)[:, np.newaxis]
        self.xsi = xsi
        self.minimize = minimize

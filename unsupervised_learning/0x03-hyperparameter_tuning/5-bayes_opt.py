#!/usr/bin/env python3
"""Defines `BayesianOptimization`."""
import numpy as np
from scipy.stats import norm
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

    def acquisition(self):
        """
        Calculates the next best sample location using the Expected
        Improvement acquisition function.

        Returns: (X_next, EI)
            X_next: A numpy.ndarray of shape (1,) representing the next
                best sample point.
            EI: A numpy.ndarray of shape (ac_samples,) containing the
                expected improvement of each potential sample.
        """
        means, standard_deviations = self.gp.predict(self.X_s)
        if self.minimize:
            best_sample = min(self.gp.Y)
            improvement = best_sample - means - self.xsi
        else:
            best_sample = max(self.gp.Y)
            improvement = means - best_sample - self.xsi

        with np.errstate(divide='ignore'):
            Z = improvement / standard_deviations
            expected_improvement = (
                improvement * norm.cdf(Z) + standard_deviations * norm.pdf(Z)
            )
            expected_improvement[standard_deviations == 0.0] = 0.0

        X_next = self.X_s[np.argmax(expected_improvement)]

        return X_next, expected_improvement

    def optimize(self, iterations=100):
        """
        Optimizes the black-box function. If the next proposed point is one
        that has already been sampled, optimization will be stopped early.

        iterations: The maximum number of iterations to perform.

        Returns: (X_opt, Y_opt)
            X_opt: A numpy.ndarray of shape (1,) representing the optimal
                point.
            Y_opt: A numpy.ndarray of shape (1,) representing the optimal
                function value.
        """
        for _ in range(iterations):
            x, _ = self.acquisition()
            if x in self.gp.X:
                break
            y = self.f(x)
            self.gp.update(x, y)

        optimum = np.argmin if self.minimize else np.argmax
        optimum_point_index = optimum(self.gp.Y)
        optimum_x = self.gp.X[optimum_point_index]
        optimum_y = self.gp.Y[optimum_point_index]
        self.gp.X = self.gp.X[:-1]

        return (optimum_x, optimum_y)

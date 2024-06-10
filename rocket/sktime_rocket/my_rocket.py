# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 14:41:04 2024

@author: Yuchen HOU
"""

import numpy as np
import pandas as pd
from rocket.sktime_rocket.rocket_kernel_function import _apply_kernels, _generate_kernels

class Rocket():
    """RandOm Convolutional KErnel Transform (ROCKET).
    Parameters
    ----------
    num_kernels : int, default=10,000
       number of random convolutional kernels.
    normalise : boolean, default True
       whether or not to normalise the input time series per instance.
    n_jobs : int, default=1
       The number of jobs to run in parallel for `transform`. ``-1`` means use all
       processors.
    random_state : None or int, optional, default = None
    """

    def __init__(self, num_kernels=10_000, normalise=True, n_jobs=1, random_state = None):
        self.num_kernels = num_kernels
        self.normalise = normalise
        self.n_jobs = n_jobs
        self.random_state = random_state if isinstance(random_state, int) else None
        super().__init__()

    def fit(self, X, y=None):
        """Generate random kernels adjusted to time series shape.

        Infers time series length and number of channels / dimensions (
        for multivariate time series) from input pandas DataFrame,
        and generates random kernels.

        Parameters
        ----------
        X : 3D np.ndarray of shape = [n_instances, n_dimensions, series_length]
            panel of time series to transform
        y : ignored argument for interface compatibility

        Returns
        -------
        self
        """
        _, self.n_columns, n_timepoints = X.shape
        self.kernels = _generate_kernels(
            n_timepoints, self.num_kernels, self.n_columns, self.random_state
        )
        return self

    def transform(self, X, y=None):
        """Transform input time series using random convolutional kernels.

        Parameters
        ----------
        X : 3D np.ndarray of shape = [n_instances, n_dimensions, series_length]
            panel of time series to transform
        y : ignored argument for interface compatibility

        Returns
        -------
        pandas DataFrame, transformed features
        """
        if self.normalise:
            X = (X - X.mean(axis=-1, keepdims=True)) / (
                X.std(axis=-1, keepdims=True) + 1e-8
            )
            
        t = pd.DataFrame(_apply_kernels(X.astype(np.float32), self.kernels))
        return t
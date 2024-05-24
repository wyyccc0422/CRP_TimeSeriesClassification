#PYTS Package Opt

from rocket.utils import apply_all_kernels,precompute_kernel_parameters,generate_kernels_optimized
import numpy as np
import pandas as pd

"""Code for RandOm Convolutional KErnel Transformation."""


"""
To reduce the transform time, we optimized the process which validate parameters passed to the methods in ROCKET class. 
Specifically:
        - Deprecated the use of `_check_params` method by directly converting variables to the desired data types.
        - Deprecated the `check_is_fitted` process which checks if the estimator is fitted by verifying the presence of fitted attributes
"""


class Rocket_Faster():
    """
    RandOm Convolutional KErnel Transformation.

    This algorithm randomly generates a great variety of convolutional kernels 
    and extracts two features for each convolution: the maximum and the proportion of positive values.

    Examples Usage:
    --------------
        >>> from faster_pyts import ROCKET_Faster
        >>> X = np.arange(100).reshape(5, 20)
        >>> rocket = ROCKET(n_kernels=10)
        >>> rocket.fit_transform(X).shape
        (5, 20)
    """

    def __init__(self, n_kernels=10000, kernel_sizes=(7, 9, 11),
                 random_state=None):
        self.n_kernels = n_kernels
        self.kernel_sizes = kernel_sizes
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit the model according to the given training data."""
        
        X = X.astype('float32') ##Replace: X = check_array(X, dtype='float64')
        n_samples, n_timestamps = X.shape
        
        kernel_sizes = np.array(self.kernel_sizes, dtype = 'int32') ##Replace: kernel_sizes, seed = self._check_params(n_timestamps)

        if self.random_state is None:
            rng = np.random.default_rng(self.random_state)
            seed = rng.integers(np.iinfo(np.uint32).max, dtype='uint64')

        # Generate the kernels
        lengths, cumsum_lengths, weights_all, biases, upper_bounds, padding_cond = precompute_kernel_parameters(self.n_kernels,
                                                                                                                n_timestamps,
                                                                                                                kernel_sizes,
                                                                                                                seed)
        self.weights_, self.length_, self.bias_, self.dilation_, self.padding_ = generate_kernels_optimized(lengths, cumsum_lengths,
                                                                                                  weights_all, biases,
                                                                                                  upper_bounds, padding_cond,
                                                                                                  self.n_kernels, kernel_sizes,
                                                                                                  n_timestamps)

        return self

    def transform(self, X):
        """
        Transform the provided data
        Returns
        -------
        X_new : DataFrame, size = (n_samples, 2 * n_kernels)
                Extracted features from the kernels.
        """

        X = X.astype('float64')
        t = pd.DataFrame(apply_all_kernels(X, self.weights_, self.length_, self.bias_, self.dilation_, self.padding_))

        return t



    #Deprecated Method
    # def _check_params(self, n_timestamps):
    #     if not isinstance(self.n_kernels, (int, np.integer)):
    #         raise TypeError("'n_kernels' must be an integer (got {})."
    #                         .format(self.n_kernels))

    #     if not isinstance(self.kernel_sizes, (list, tuple, np.ndarray)):
    #         raise TypeError("'kernel_sizes' must be a list, a tuple or "
    #                         "an array (got {}).".format(self.kernel_sizes))
    #     kernel_sizes = check_array(self.kernel_sizes, ensure_2d=False,
    #                                dtype='int64', accept_large_sparse=False)
    #     if not np.all(1 <= kernel_sizes):
    #         raise ValueError("All the values in 'kernel_sizes' must be "
    #                          "greater than or equal to 1 ({} < 1)."
    #                          .format(kernel_sizes.min()))
    #     if not np.all(kernel_sizes <= n_timestamps):
    #         raise ValueError("All the values in 'kernel_sizes' must be lower "
    #                          "than or equal to 'n_timestamps' ({} > {})."
    #                          .format(kernel_sizes.max(), n_timestamps))

    #     rng = check_random_state(self.random_state)
    #     seed = rng.randint(np.iinfo(np.uint32).max, dtype='u8')

    #     return kernel_sizes, seed

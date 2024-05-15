#PYTS Package Opt

from rocket.utils import apply_all_kernels,precompute_kernel_parameters,generate_kernels_optimized
import numpy as np
import pandas as pd

"""Code for RandOm Convolutional KErnel Transformation."""

class Rocket_Faster():
    """RandOm Convolutional KErnel Transformation."""

    def __init__(self, n_kernels=10000, kernel_sizes=(7, 9, 11),
                 random_state=None):
        self.n_kernels = n_kernels
        self.kernel_sizes = kernel_sizes
        self.random_state = random_state

    def fit(self, X, y=None):
        X = X.astype('float32') 
        n_samples, n_timestamps = X.shape

        kernel_sizes = np.array(self.kernel_sizes, dtype = 'int32')

        # Faster method: 
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
        X = X.astype('float64')
        t = pd.DataFrame(apply_all_kernels(X, self.weights_, self.length_, self.bias_, self.dilation_, self.padding_))

        return t




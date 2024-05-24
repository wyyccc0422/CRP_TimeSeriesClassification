"""Code for RandOm Convolutional KErnel Transformation."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause
from numba import njit, prange
import numpy as np


#Original Function
def precompute_kernel_parameters(n_kernels, n_timestamps, kernel_sizes, seed):
    np.random.seed(seed)

    lengths = np.random.choice(kernel_sizes, size=n_kernels)
    cumsum_lengths = np.concatenate((np.array([0]), np.cumsum(lengths)))
    weights_all = np.random.randn(cumsum_lengths[-1])
    biases = np.random.uniform(-1, 1, size=n_kernels)
    upper_bounds = np.log2(np.floor_divide(n_timestamps - 1, lengths - 1))
    padding_cond = np.random.randint(0, 2, size=n_kernels).astype(np.bool_)

    return lengths, cumsum_lengths, weights_all, biases, upper_bounds, padding_cond


#Optimised Kernel Generation Function
@njit()
def generate_kernels_optimized(lengths, cumsum_lengths, weights_all, biases, upper_bounds, padding_cond, n_kernels, kernel_sizes, n_timestamps):
    weights = np.zeros((n_kernels, np.int64(np.max(kernel_sizes))))
    dilations = np.zeros(n_kernels)
    paddings = np.zeros(n_kernels)

    for i in prange(n_kernels):
        start, end = cumsum_lengths[i], cumsum_lengths[i+1]
        kernel_length = lengths[i]
        weights[i, :kernel_length] = weights_all[start:end] - np.mean(weights_all[start:end])
        dilations[i] = np.floor(np.power(2, np.random.uniform(0, upper_bounds[i])))

        if padding_cond[i]:
            paddings[i] = np.floor_divide((kernel_length - 1) * dilations[i], 2)

    return weights, lengths, biases, dilations.astype('int64'), paddings.astype('int64')



"""
Based on the pyts ROCKET source code, we created a new `apply_all_kernels` function whiich replaced the following two functions:
    - apply_one_kernel_one_sample
    - apply_all_kernels

Specifically by mergering these two functions into one, we:
        - Optimised the inner loop which applies the convolutional kernel to each time series
        - Parallelize the outer loop while keeping the inner loop sequential.
"""
#Adjusted Function
@njit(parallel=True, fastmath=True)
def apply_all_kernels(X, weights, lengths, biases, dilations, paddings):
    """
    Apply one kernel to a data set of time series.
    
    Parameters
    ----------
    X : array, shape = (n_samples, n_timestamps)
        Input data.

    weights : array, shape = (n_kernels, max(kernel_sizes))
        Weights of the kernels. Zero padding values are added.

    lengths : array, shape = (n_kernels,)
        Length of each kernel.

    biases : array, shape = (n_kernels,)
        Bias of each kernel.

    dilations : array, shape = (n_kernels,)
        Dilation of each kernel.

    paddings : array, shape = (n_kernels,)
        Padding of each kernel.

    Returns
    -------
    X_new : array, shape = (n_samples, 2 * n_kernels)
        Extracted features (maximum values and ppv)
    """
    n_samples, n_timestamps = X.shape
    n_kernels = lengths.size
    X_new = np.empty((n_samples, 2 * n_kernels))

    for i in prange(n_samples):
        for j in prange(n_kernels):
            # Compute padded x
            n_conv = n_timestamps - ((lengths[j] - 1) * dilations[j]) + (2 * paddings[j])

            if paddings[j] > 0:
                x_pad = np.zeros(n_timestamps + 2 * paddings[j])
                x_pad[paddings[j]:-paddings[j]] = X[i]
            else:
                x_pad = X[i]

            x_conv = np.zeros(n_conv)
            conv_indices = np.arange(0, n_conv - (lengths[j] - 1) * dilations[j], dilations[j])
            for k in prange(lengths[j]):
                # # Calculate the indices for convolution
                for l in range(0, n_conv - (lengths[j] - 1) * dilations[j], dilations[j]):
                    x_conv[l+k] += weights[j][k] * x_pad[l + k]
            x_conv += biases[j]
            # print(x_conv)
            # Store the features: maximum and proportion of positive values
            X_new[i, 2 * j] = np.max(x_conv)
            X_new[i, 2 * j + 1] = np.mean(x_conv > 0)

    return X_new



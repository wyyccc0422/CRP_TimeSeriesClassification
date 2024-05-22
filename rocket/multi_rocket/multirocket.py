#The MultiROCKET implemented here is created by: 

#Chang Wei Tan, Angus Dempster, Christoph Bergmeir, Geoffrey I Webb

#MultiRocket: Multiple pooling operators and transformations for fast and effective time series classification
# https://arxiv.org/abs/2102.00457


"""
Given that we are focusing solely on the transformer, we've streamlined the original MultiROCKET Class. 
Specifically:
        - Removed the verbose logging, classifier,and prediction.
        - Retained only core feature generation functions.
        - Modified the`fit` method to fit the kernels without needing to specify the target labels(y_train).
        - Added a `transform` method which separates the transformation process from the original fitting method.
        - Specified `num_kernels` directly in the constructor and used it to compute the number of features.
"""

import numpy as np
from rocket.multi_rocket.utils import fit,transform


# Streamlined MultiROCKET Class
class MultiRocket:

    def __init__(
            self,
            num_kernels=10000
    ):
        self.name = "MultiRocket"

        self.base_parameters = None
        self.diff1_parameters = None

        self.n_features_per_kernel = 4
        self.num_kernels = num_kernels
        self.num_features = self.num_kernels *self.n_features_per_kernel 

    def fit(self,x_train):

        xx = np.diff(x_train, 1) # compute the first-order differences of the input array 

        self.base_parameters = fit(
            x_train,
            num_features=self.num_kernels
            )

        self.diff1_parameters = fit(
            xx,
            num_features=self.num_kernels
        )
        return self 

    def transform(self, x_train):
        xx = np.diff(x_train, 1)
        x_train_transform = transform(
            x_train, xx,
            self.base_parameters, self.diff1_parameters,
            self.n_features_per_kernel
        )
        
        x_train_transform = np.nan_to_num(x_train_transform) #Replace NaN with zero and infinity with large finite numbers 

        return x_train_transform


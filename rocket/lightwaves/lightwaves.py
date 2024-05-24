import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif, VarianceThreshold

from rocket.lightwaves.lightwavesl1l2_functions import _generate_first_phase_kernels, _apply_2layer_kernels
from rocket.lightwaves.lightwaves_utils import ScalePerChannel, anova_feature_selection, mrmr_feature_selection, ScalePerChannelTrain, \
    ckd_to_kernels, get_fixed_candidate_kernels, get_ckd_matrix_with_features


def transform(X, matrix, feat_mask, candidate_kernels, dilations):
    """
    Transform input array to LightWaveS features
    :param X: The input timeseries array of dimension (samples,channels,timesteps)
    :param matrix: A channel-kernel-dilation 2d array of dimensions (n_kernels,3)
    :param feat_mask: Feature mask of LightWaveS of dimension (n_kernels,features_number). Describes which features to keep from each kernel application
    :param candidate_kernels: The set of base kernels used by LightWaveS
    :param dilations: The set of base dilations used by LightWaveS
    :return: Transformed array of dimensions (samples,features)
    """
    kernels = ckd_to_kernels(matrix, candidate_kernels, dilations)
    feats = _apply_2layer_kernels(X, kernels)
    return feats[:, feat_mask]


"""
In the public repository of LightWaveS (https://github.com/lpphd/lightwaves/tree/main),
The author does not define the LightWaveS as a Class Method, 
Thus, for the purpose of this project, 
& to allow any function that takes a ROCKET object as input to call `fit` and `transform` methods uniformly, regardless of the specific ROCKET class used,
We converted the python script into the object-oriented programming (LightWaveS Class).

Specifically:
        - Reference python scrpt we used: `lightwaves_l1l2_UCR_example.py` and `lightwaves_l1l2_MAFAULDA_example.py` 
          in which the authors of lightwaves repository showed examples of applying lightwaves on two datasets. 
        - Removed the use of sample size as for our project we are using the whole dataset. --> No need to take the sample
        - Removed the use of MPI because it can't be imported
       
"""


class LightWaveS:
    def __init__(self, seed=0, final_num_feat=500, max_dilation=32):
        self.seed = seed
        self.final_num_feat = final_num_feat  ##Final number of features
        self.max_dilation = max_dilation
        self.features_number = 8  ## 4 features per scattering level
        self.pre_final_feat_num = 3 * final_num_feat  ## Pool of features selected on each node, before the final selection.
        self.dilations = np.arange(0, np.log2(max_dilation) + 1).astype(np.int32)
        self.n_dilations = self.dilations.size

    def fit(self, X_train, y_train=None,normalized = True):
        np.random.seed(self.seed)

        if not normalized:
            X_train = ScalePerChannelTrain(X_train)

        total_num_channels = X_train.shape[1]

        # Process all channels
        my_channels = np.arange(total_num_channels)

        # Generate candidate kernels
        self.candidate_kernels = get_fixed_candidate_kernels()
        first_phase_kernels = _generate_first_phase_kernels(total_num_channels, self.candidate_kernels, self.dilations, self.seed)

        # Apply first-phase kernels to the training data
        transform_features = _apply_2layer_kernels(X_train, first_phase_kernels)

        # Select best features using ANOVA
        sel_feat_idces, sel_feat_scores = anova_feature_selection(
            transform_features.reshape((transform_features.shape[0], -1)), y_train,
            self.pre_final_feat_num
        )

        # Transform selected feature indices
        ckdf = get_ckd_matrix_with_features(sel_feat_idces, total_num_channels, len(self.candidate_kernels), self.n_dilations, self.features_number)

        # Select unique kernels and create feature mask
        unique_ckdf = np.unique(ckdf[:, :-1], axis=0)
        cand_kernels = ckd_to_kernels(unique_ckdf, self.candidate_kernels, self.dilations)
        feat_mask = np.zeros((unique_ckdf.shape[0], self.features_number), dtype=bool)
        sel_feat_per_k = list(pd.DataFrame(ckdf).groupby([0, 1, 2])[3].apply(list))
        for i in range(feat_mask.shape[0]):
            feat_mask[i, sel_feat_per_k[i]] = True

        # Apply kernels to the training data and mask features
        cand_feats = _apply_2layer_kernels(X_train.astype(np.float32), cand_kernels)[:, feat_mask]

        # Select best features using mrmr
        global_sel_feats_p2_idces, _ = anova_feature_selection(cand_feats, y_train, self.final_num_feat)

        # Finalize kernel matrix and feature mask
        ckdf = ckdf[global_sel_feats_p2_idces, :]
        self.kernel_matrix_final = np.unique(ckdf[:, :-1], axis=0)
        self.feat_mask = np.zeros((self.kernel_matrix_final.shape[0], self.features_number), dtype=bool)
        sel_feat_per_k = list(pd.DataFrame(ckdf).groupby([0, 1, 2])[3].apply(list))
        for i in range(self.feat_mask.shape[0]):
            self.feat_mask[i, sel_feat_per_k[i]] = True

        return self

    def transform(self, X,normalized=True):
        if not normalized:
            X = ScalePerChannelTrain(X)
        train_tr = transform(X.astype(np.float32), self.kernel_matrix_final, self.feat_mask, self.candidate_kernels, self.dilations)
        return train_tr

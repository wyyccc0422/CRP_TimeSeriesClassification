import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif, VarianceThreshold
from mpi4py import MPI

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
        - Removed the `get samples` as for our project we are using the whole dataset. --> No need to take the sample
       
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

        ## Depending on number of channels and distribute nodes, the MPI ranks may change during execution
        self.orig_comm = MPI.COMM_WORLD
        self.orig_rank = self.orig_comm.Get_rank()
        self.orig_n_nodes = self.orig_comm.Get_size()

    def fit(self, X_train, y_train=None,normalized = True):
        np.random.seed(self.seed)
        VERBOSE = 0
        self.orig_comm.Barrier()
        for seed in range(1):
            self.orig_comm.Barrier()
            np.random.seed(seed)

        total_num_channels = X_train.shape[1]

        if total_num_channels < self.orig_n_nodes:
            if self.orig_rank == 0:
                if VERBOSE:
                    print("Number of channels is smaller than number of nodes, reducing COMM to subset.")
            comm = self.orig_comm.Create_group(self.orig_comm.group.Incl(np.arange(total_num_channels).tolist()))
        else:
            comm = self.orig_comm

        ## Split channels across MPI nodes
        channel_distribution = np.array_split(np.arange(total_num_channels), self.orig_n_nodes)

        ## Get channels of this node
        my_channels = channel_distribution[self.orig_rank]

        if self.orig_rank < total_num_channels:

            rank = comm.Get_rank()
            n_nodes = comm.Get_size()

            if not normalized:
                X_train = ScalePerChannelTrain(X_train)
            X_train = X_train.astype(np.float32)

            # Generate candidate kernels
            self.candidate_kernels = get_fixed_candidate_kernels()
            if rank == 0:
                if VERBOSE:
                    print(dataset, seed)
                    print(candidate_kernels.shape[0] * n_dilations * total_num_channels)
            # Apply first-phase kernels to the training data
            first_phase_kernels = _generate_first_phase_kernels(total_num_channels, self.candidate_kernels, self.dilations, self.seed)

            #Transform train set
            transform_features = _apply_2layer_kernels(X_train, first_phase_kernels)

            # Select best features using ANOVA
            sel_feat_idces, sel_feat_scores = anova_feature_selection(
                transform_features.reshape((transform_features.shape[0], -1)), y_train,
                self.pre_final_feat_num
            )

            ##Send feature scores to main node for comparison
            ##First send number of features to main node
            feat_count = np.array(sel_feat_idces.size).reshape((1, 1))
            feat_count_recvbuf = None
            if rank == 0:
                feat_count_recvbuf = np.empty([n_nodes], dtype='int')
            comm.Gather(feat_count, feat_count_recvbuf, root=0)


            ## Then send actual scores to main node
            displ = None
            feat_scores_recvbuf = None
            counts = None
            feat_score_sendbuf = sel_feat_scores.flatten()
            if rank == 0:
                displ = np.hstack((0, feat_count_recvbuf.flatten())).cumsum()[:-1]
                feat_scores_recvbuf = np.empty((feat_count_recvbuf.sum()), dtype=np.float32)
                counts = feat_count_recvbuf
            
            comm.Gatherv(feat_score_sendbuf, [feat_scores_recvbuf, counts, displ, MPI.FLOAT], root=0)

            ## Main node sorts scores and sends back to each node how many (if any) of its top features to send
            if rank == 0:
                score_src_idces = []
                for i in range(n_nodes):
                    score_src_idces.extend([i] * feat_count_recvbuf[i])
                score_src_idces = np.array(score_src_idces)

                top_score_src_count = np.bincount(score_src_idces[np.argsort(feat_scores_recvbuf.flatten())[::-1]][
                                                :self.pre_final_feat_num], minlength=n_nodes).astype(np.int32)
            else:
                top_score_src_count = np.empty(n_nodes, dtype=np.int32)
            
            comm.Bcast(top_score_src_count, root=0)

            ## On each node, select top features (if any)
            sel_feat_idces = np.sort(sel_feat_idces[np.argsort(sel_feat_scores)[::-1]][:top_score_src_count[rank]])

            if (top_score_src_count == 0).any():
                if orig_rank == 0 and VERBOSE == 1:
                    print("Some nodes have 0 CKD selected, reducing COMM to subset.")

                new_comm = comm.Create_group(comm.group.Incl(np.where(top_score_src_count != 0)[0].tolist()))
            else:
                new_comm = comm

            if top_score_src_count[rank] > 0:
                rank = new_comm.Get_rank()
                n_nodes = new_comm.Get_size()
                
                ## Transform node feature indices to final format of channel-kernel-dilation-feature
                ckdf = get_ckd_matrix_with_features(sel_feat_idces, total_num_channels, len(self.candidate_kernels), self.n_dilations, self.features_number)
                
                ckdf[:, 0] = my_channels[ckdf[:, 0]]

                ##Send kernel matrices to main node for second comparison
                displ = None
                ckdf_recvbuf = None
                counts = None
                feat_sendbuf = ckdf.flatten()
                if rank == 0:
                    displ = np.hstack((0, top_score_src_count[top_score_src_count != 0].flatten())).cumsum()[:-1] * 4
                    ckdf_recvbuf = np.empty((4 * top_score_src_count.sum()), dtype=np.int32)
                    counts = top_score_src_count[top_score_src_count != 0] * 4

                new_comm.Gatherv(feat_sendbuf, [ckdf_recvbuf, counts, displ, MPI.INT], root=0)

                if rank == 0:
                    ckdf_recvbuf = ckdf_recvbuf.reshape((-1, 4))
        
                    ## On main node, keep unique kernels (some kernels may give more than 1 feature)
                    unique_ckdf = np.unique(ckdf[:, :-1], axis=0)

                    ## Create kernel matrix and feature mask
                    cand_kernels = ckd_to_kernels(unique_ckdf, self.candidate_kernels, self.dilations)
                    feat_mask = np.zeros((unique_ckdf.shape[0], self.features_number), dtype=bool)
                    sel_feat_per_k = list(pd.DataFrame(ckdf).groupby([0, 1, 2])[3].apply(list))
                    for i in range(feat_mask.shape[0]):
                        feat_mask[i, sel_feat_per_k[i]] = True

                    ## Transform train set with list of received kernels
                    cand_feats = _apply_2layer_kernels(X_train.astype(np.float32), cand_kernels)[:, feat_mask]

                    # Select best features using mrmr
                    global_sel_feats_p2_idces, _, _ =  mrmr_feature_selection(cand_feats,
                                                        y_train,
                                                        self.final_num_feat)

                    ## Keep best features from the previously received kernel set, generate final kernel matrix and feature mask
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

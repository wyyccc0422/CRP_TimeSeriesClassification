import pandas as pd
import numpy as np
import statistics
from scipy.stats import norm
from utils import dfwellgr,window
import torch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import scipy as sp



    


def fit_gaussian_process(df_tops, confidence_level=0.96):
    """
    Fits a Gaussian process regression model for depth measurements and find the confidence intervals.

    Returns:
    -----------
       intervals [dict]: A dictionary containing confidence intervals for each marker.
    """
    intervals ={}
    for marker in df_tops.columns:
        y = df_tops[marker].dropna().values
        X = np.atleast_2d(np.arange(len(y))).T	# creates a 2d column vectors of indices (based on len(y)) // each represent the position of each depth measurement in the array
        
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-4, 1e2))

        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-10, normalize_y=True)
        gp.fit(X, y)
        
        # Predict with confidence intervals
        y_pred, sigma = gp.predict(X, return_std=True)
        confidence_interval = sigma * np.sqrt(2) * sp.special.erfinv(confidence_level)
        
        lower_b = np.min(y_pred - confidence_interval)
        upper_b = np.max( y_pred + confidence_interval)

        intervals[marker] = (lower_b,upper_b)

    return intervals

def fit_lognormal_process(df_tops, confidence_level=0.96):
    """
    Fits a Gaussian process regression model for depth measurements and find the confidence intervals.
    Using log-normal distribution.
   
    Returns:
    -----------
       intervals [dict]: A dictionary containing confidence intervals for each marker.
    """
    intervals ={}
    for marker in df_tops.columns:
        ## transform to log
        y = np.log(df_tops[marker].dropna().values)
        X = np.atleast_2d(np.arange(len(y))).T
        
        ## Kernel for Gaussian Process (because logNORMAL)
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-4, 1e2))
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-10, normalize_y=True)
        gp.fit(X, y)
        
        # Predict on the training data to get mean and var
        y_pred, sigma = gp.predict(X, return_std=True)
        
        # log scaled conf interval 
        ci_low = y_pred - norm.ppf((1 + confidence_level) / 2) * sigma
        ci_high = y_pred + norm.ppf((1 + confidence_level) / 2) * sigma
        
        
        # scaled back conf interval 
        ci_low_original = np.exp(ci_low)
        ci_high_original = np.exp(ci_high)

        intervals[marker] = (np.min(ci_low_original),np.max(ci_high_original))

    return intervals


def apply_process(df_tops,confidence_level=0.96,log=False):

    if log:
        intervals = fit_lognormal_process(df_tops, confidence_level)
    else:
        intervals = fit_gaussian_process(df_tops, confidence_level)
    for k,(v1,v2) in intervals.items():
        print(k,round(v1,0),round(v2,0))
    return intervals


def predict_s2s(test_well,model):
        """Prediction with s2s Model"""
        input_tensor = torch.tensor(test_well,dtype=torch.float)
        batch_size = 64 
        num_batches = int(np.ceil(input_tensor.shape[0] / batch_size))
        predictions = []

        for i in range(num_batches):
            batch = input_tensor[i * batch_size:(i + 1) * batch_size]
            batch_predictions = model(batch)
            predictions.append(batch_predictions)

        test_prob = torch.cat(predictions, dim=0)
        test_prob = test_prob.detach().numpy() 
        return test_prob 



def get_markers_rocket_order_with_constraint(well,
                                             f_mean, 
                                             f_std, 
                                             intervals,
                                             df_test_log, 
                                             pred_column, 
                                             wsize, 
                                             input_variable, 
                                             s2s = False,
                                             model = None, 
                                             xgb = True,
                                             rocket = None, 
                                             classifier_xgb = None, 
                                             classifier = None,
                                             constraint = True,
                                            ):
    """ 
    Predict marker depths for one well with weighted blending of original and constrained predictions using Empirical CDF
    
    Parameters
    ----------
        f_mean [array]: Mean values of X_train used for normalization.
        f_std [array] : Standard deviation values of X_train used for normalization.
        intervals : lower
        df_test_log [DataFrame]: DataFrame containing test data.
        well : index of the well being analyzed.
        pred_column [list]: Predicted column names >> ["MARCEL","SYLVAIN","CONRAD"]
        wsize [int]: Window size for data processing.
        input_variable [list]: input variables used for prediction >> ["GR"]
                               only consider uni-varaite case for now
        classifier_xgb : XGBoost classifier for prediction
        classifier : Logistic classifier for prediction
        df_tops [DataFrame]: DataFrame containing training data with marker columns.
        xgb [bool]: whether XGBoost is used.
        alpha [float]: Weight for blending. Default is 0.5 for equal weighting.
        confidence_level [float]: Confidence level for empirical CDF intervals. Default is 0.96.
    
    Returns
    -------
        pred_m [list]: Predicted depths for each marker 
        df_wm [DataFrame]: DataFrame containing predicted probabilities for all markers.
    """
    #=========Transfer Testing Data to Sequence of GR For Prediction=============#

    df_gr = dfwellgr(well, df_test_log, input_variable)
    well_seq, dep_seq = window(df_gr, wsize)
    

    #==================ROKECT OR S2S MODEL================#
    if s2s: 
        test_well = (well_seq - f_mean) / f_std   ## Normalize 'per feature'
        test_prob = predict_s2s(test_well,model)
    else:  ## DO ROCKET TRANSFORM 
        try:
            test_well = rocket.transform(well_seq) 
        except Exception as e:
            well_seq = well_seq.reshape(well_seq.shape[0], -1)
            test_well = rocket.transform(well_seq)
        
        test_well = (test_well - f_mean) / f_std  ## Normalize 'per feature'
        if xgb:
            test_prob = classifier_xgb.predict_proba(test_well)
        else:
            test_prob = classifier.predict_proba(test_well)

    #==================Marker Depth Extraction================#
    df_wm = pd.DataFrame(test_prob, columns=pred_column)
    df_wm['Depth'] = dep_seq

    if 'Depth' not in df_gr.columns:
        df_gr['Depth'] = df_gr.index

    df_wm = df_gr.merge(df_wm, how = 'inner', left_on = 'Depth', right_on = 'Depth')
    df_wmn = df_wm

    #=========Predict Depth For Each Marker with Weighted Blending=============#
    

    #Normal Prediction
    pred_m = []
    for top in pred_column:
        if top != 'None':
            md = df_wmn[df_wmn[top] == df_wmn[top].max()].Depth
            ym = statistics.median(md)
            pred_m.append(ym)

    #Constraint1: Gaussian
    #Constraint2: Next Marker must be deeper than the previous#
    if constraint:
        pred_m = []
        previous_depth = 0
        # Get Gaussian Process intervals
        for top in pred_column:
            if top != 'None':
                md = df_wmn[df_wmn[top] == df_wmn[top].max()].Depth
                ym = statistics.median(md)

                # Retrieve the interval for the current marker
                lower_bound, upper_bound = intervals[top.upper()] 
                # print(f"Interval For {top} is: {lower_bound,upper_bound}")
                a = 1
                while ym < lower_bound or ym>upper_bound or ym<previous_depth:
                    a+=1
                    md = df_wmn[df_wmn[top] == df_wmn[top].nlargest(a).iloc[-1]].Depth
                    ym = statistics.median(md)
                
                previous_depth = ym
                pred_m.append(ym)
        return pred_m, df_wm
    else:
        return pred_m, df_wm

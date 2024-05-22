import pandas as pd
import numpy as np
import statistics
from scipy.stats import norm
from utils import dfwellgr,window
import torch

def get_empirical_cdf_intervals( df_tops, confidence_level=0.96):
    intervals = {}
    for marker in df_tops.columns:
        depths = df_tops[marker].dropna()
        lower_bound = np.percentile(depths, (1 - confidence_level) / 2 * 100)
        upper_bound = np.percentile(depths, (1 + confidence_level) / 2 * 100)
        intervals[marker] = (lower_bound, upper_bound)
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

def is_increasing(lst):
    """Check if the Final Prediction is in increasind order """
    for i in range(len(lst) - 1):
        if lst[i] >= lst[i + 1]:
            return False
    return True

def get_markers_rocket_order_with_constraint(f_mean, 
                                             f_std, 
                                             df_tops,
                                             df_test_log, 
                                             well, 
                                             pred_column, 
                                             wsize, 
                                             input_variable, 
                                             s2s = False,
                                             model = None, 
                                             xgb = True,
                                             rocket = None, 
                                             classifier_xgb = None, 
                                             classifier = None,
                                             alpha=0.5, 
                                             confidence_level=0.96):
    """ 
    Predict marker depths for one well with weighted blending of original and constrained predictions using Empirical CDF
    
    Parameters
    ----------
        f_mean [array]: Mean values of X_train used for normalization.
        f_std [array] : Standard deviation values of X_train used for normalization.
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
    
    # Get empirical CDF intervals
    empirical_intervals = get_empirical_cdf_intervals(df_tops, confidence_level)
    
    #Constraint 1: Gaussian#
    pred_m = []
    for top in pred_column:
        if top != 'None':
            md = df_wmn[df_wmn[top] == df_wmn[top].max()].Depth
            ym_original = statistics.median(md)

            
            # Retrieve the empirical interval for the current marker
            lower_bound, upper_bound = empirical_intervals[top.upper()] 

            # Adjust ym to fall within the empirical interval
            ym_constrained = np.clip(ym_original,lower_bound,upper_bound)
            ym_blended = alpha * ym_original + (1 - alpha) * ym_constrained
            
            previous_depth = ym_blended
            pred_m.append(ym_blended)
    

    #Constraint 2: Next Marker must be deeper than the previous#
    #This constarint is triggered only when the the depth order is wrong.#

    if is_increasing(pred_m):  ## Check if pred_m is an increasing list
        return pred_m, df_wm
    else: 
        previous_depth = pred_m[0]
        for i,top in enumerate(pred_column):
            if i>1:
                md = df_wmn[df_wmn[top] == df_wmn[top].max()].Depth
                ym_original = statistics.median(md)
                lower_bound, upper_bound = empirical_intervals[top.upper()] 

                a = 1
                while ym_original < previous_depth or ym_original>upper_bound:
                    a+=1
                    md = df_wmn[df_wmn[top] == df_wmn[top].nlargest(a).iloc[-1]].Depth
                    ym_original = statistics.median(md)

                ym_constrained = np.clip(ym_original,lower_bound,upper_bound)
                ym_blended = alpha * ym_original + (1 - alpha) * ym_constrained
        
                previous_depth = ym_blended
                pred_m[i-1] = ym_blended

        return pred_m, df_wm
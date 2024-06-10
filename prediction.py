import pandas as pd
import numpy as np
import statistics
from scipy.stats import norm
from utils import dfwellgr,window
import torch
import time


from DistributionProcess.GausianDistribution import fit_gaussian_process,fit_lognormal_process
from DistributionProcess.StableDistribution import fit_stable_distribution,fit_stable_distribution_fft,fit_cauchy_distribution, fit_cauchy_distribution_fft

#Apply Different ways of Stable Distribution 
def apply_fit_distribution(df_tops,well_array,method='stable',confidence_level = 0.95):
    start = time.time()
    intervals ={}
    for i in range(0,len(df_tops.columns)): 
        top_well_list = well_array[i][0] 
        marker = df_tops.columns[i] 
        tops = df_tops[df_tops.index.isin(top_well_list)]

        if method == 'gausian':
            (lower_b, upper_b) = fit_gaussian_process(df_tops,marker,confidence_level)
        elif method == 'neo-log':
            (lower_b, upper_b) = fit_stable_distribution(df_tops,marker,confidence_level)
        elif method =='fft':
            (lower_b, upper_b) = fit_stable_distribution_fft(df_tops,marker,confidence_level)
        elif method == 'cauchy':
            (lower_b, upper_b) = fit_cauchy_distribution(df_tops,marker,confidence_level)
        elif method =='cauchy-fft':
            (lower_b, upper_b) = fit_cauchy_distribution_fft(df_tops,marker,confidence_level)
        else:
            print("Method Unknown, available methods are ['gaussian','stable', 'fft', 'caucy']")

        intervals[marker] = (lower_b, upper_b)
    
    ext = time.time() - start
    key_width = max(len(k) for k in intervals.keys()) + 2
    value_width = 5

    print(f"\033[1m{method} Stable Distribution 🍺 Total Time {ext}\033[0m")
    for k,(v1,v2) in intervals.items():
        print(f"{k:<{key_width}} {round(v1, 0):>{value_width}} {round(v2, 0):>{value_width}} ({round(v2-v1, 0)}m)")
        
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

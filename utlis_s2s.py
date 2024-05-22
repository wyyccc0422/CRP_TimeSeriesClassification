
from utils import dfwellgr,window
import torch
import numpy as np
import pandas as pd
import statistics

def get_markers_rocket_order(f_mean,f_std,df_test_log, well, pred_column, wsize, input_variable, model):
    """ 
    Predict marker depths for one well without any constraints
    
    Parameters
    ----------
        f_mean [array]:  Mean values of X_train used for normalization.
        f_std [array] :  Standard deviation values of X_train used for normalization.
        df_test_log [DataFrame]: DataFrame containing test data.
        well : index of the well being analyzed.
        pred_column [list]: Predicted column names >> ["MARCEL","SYLVAIN","CONRAD"]
        wsize [int]: Window size for data processing.
        input_variable [list]: input variables used for prediction >> ["GR"]
                               only consider uni-varaite case for now
        model: trained s2s model
   
    Returns
    -------
        pred_m [list]: Predicted depths for each marker 
        df_wm [DataFrame]: DataFrame containing predicted probabilities for all markers.
    """

    df_gr = dfwellgr(well, df_test_log, input_variable)
    well_seq, dep_seq = window(df_gr, wsize)
    
    # normalize 'per feature'
    test_well = (well_seq - f_mean) / f_std

    #Prediction with s2s Model
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
    # Marker Depth Extraction:
    df_wm = pd.DataFrame(test_prob, columns=pred_column)
    df_wm['Depth'] = dep_seq

    if 'Depth' not in df_gr.columns:
        df_gr['Depth'] = df_gr.index

    df_wm = df_gr.merge(df_wm, how = 'inner', left_on = 'Depth', right_on = 'Depth')
    pred_m = []
    df_wmn = df_wm

    #=========Prediction Depth For Each Marker=============#

    for top in pred_column:
        if top != 'None':
            md = df_wmn[df_wmn[top] == df_wmn[top].max()].Depth
            ym = statistics.median(md)
            pred_m.append(ym)

    return pred_m, df_wm
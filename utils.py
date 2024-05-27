import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import seaborn as sns
import os
import statistics
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

def dfwellgr(well_name, df_log, input_variable):
  """
   Returning a DataFrame that contains average GR values at each depth for one well_name.

   Parameters:
   -----------
    well_name: index of a well
    df_log: DataFrame, [wellName,Latitude,Longitude,DEPTH,GR]
    input_variable: ['GR']
   Return:
   -----------
    df_wgr: DataFrame,
            column = GR(average GR of each depth)
            index = DEPTH
  """

  df_wgr = df_log.where(df_log['wellName'] == well_name).dropna().copy()
  df_wgr.loc[df_wgr['GR'] == -1, 'GR'] = df_wgr['GR'].mean()  #replace GR=-1 with average GR
  df_wgr['DEPTH'] = df_wgr['DEPTH'].astype('int')  #set depth to integer
  df_wgr = df_wgr.groupby('DEPTH').mean()  #group by depth 
  df_wgr = df_wgr[input_variable]

  return df_wgr


def marker_ssig(well_name, df_wgr, marker, wsize):
  """
  extract segments of gamma ray data centered around a specific marker depth within a well's log data

  Parameters:
  -----------
     well_name: index of a well
     df_wgr: DataFrame, column = GR(average GR of each depth), index = DEPTH
     marker: marker depth
     wsize: window size

  Returns:
  -----------
    np_mseg: NumPy Array,shape(4, wsize))
             segments of gamma ray data centered around the marker depth.
  """

  hd = int(wsize*0.5)
  std = 2
  md = marker
  dseq = np.arange(int(md - (hd + std)), int(md + (hd + std)))
  seq = df_wgr[df_wgr.index.isin(dseq)].to_numpy() #create the series with gamma rays
 
  mseg = [np.transpose(seq[i: i + wsize]) for i in range(len(seq) - wsize + 1)]

  np_mseg = np.array(mseg)

  return np_mseg


def extract_signature_Xy(logs,tops,well_list,input_variable, wsize):
  """
  extracts signatures from the log data around each marker depth
  and creates corresponding labels for each signature
  
  Returning the dataset for machine learning

   Parameters:
   -----------
    logs: DataFrame, [wellName,Latitude,Longitude,DEPTH,GR]
    tops: DataFrame,  depth of each marker for each well
    well_list: np.array,  index for each well(unique)
    input_variable: ['GR']
    wsize: window size
  Returns:
  -----------
    X: Array, extracted signatures
    y: Array, labels
  """

  if len(well_list) == 0:
          well_list = list(tops.index)

  wsize = wsize
  half_size = int(wsize*0.5)

  X=[]
  y=[]

  progress_bar = tqdm(total=len(tops.index), desc='Processing wells')
  #loop over each well
  for well in tops.index: 
      progress_bar.set_description(f'Processing {well}')

      df_wgr = dfwellgr(well, logs, input_variable) # get average GR values at each depth for one well.

      #pass trough the different markers to get their signature
      for i in range(0,len(tops.columns)): 

          top = tops.columns[i] 
          top_well_list = well_list[i][0] 

          #assign the well list for the marker and create good signatures for the marker
          if well in top_well_list:

              marker = tops.loc[well][i] #marker depth
              np_mseg = marker_ssig(well, df_wgr, marker, wsize)
              y_seq = np.full(len(np_mseg), i+1) # Create a 1xlen(np_mseg) array filled with the value i+1
              y.append(y_seq)
              X.append(np_mseg)

      #Generate data series for non marker labels
      wtop = tops.loc[well].values
      rd = np.random.randint(low=1000, high=max(tops.loc[well]))
      while (rd in wtop):
        rd = np.random.randint(low=1000, high=max(tops.loc[well])) #choose a rd that is not in wtop(non marker)

      np_mseg = marker_ssig(well, df_wgr, rd, wsize)
      
      y_seq = np.full(len(np_mseg), 0) #assign label 0 to non marker
      y.append(y_seq)
      X.append(np_mseg)

      progress_bar.update(1)

  progress_bar.close()

  #Save data to local
  directory = 'prepared_data'
  if not os.path.exists(directory):
      os.makedirs(directory)

  # Assuming X is your NumPy array
  np.save(os.path.join(directory, f'X_{wsize}.npy'), np.concatenate(X))
  np.save(os.path.join(directory, f'y_{wsize}.npy'), np.concatenate(y))
  print("Data Saved Successfully")

  return np.concatenate(X), np.concatenate(y)


# ======================================FUNCTION FOR TESTING====================================#
def window(df_wgr, wsize):
  """
  extract segments of data from a DataFrame (df_wgr) without specific reference to marker depths
  (same as marker_ssig but don't need marker reference as it's used for testing data)
  
   Parameters:
   -----------
    df_wgr: DataFrame
            column = GR(average GR of each depth)
            index = DEPTH
    wsize: window size
  Returns:
  -----------
    np.array(mseg): segments of gamma ray data, each segment of size 'wsize'
    dep_list: List, list of depth values corresponding to the center of each segment
  """

  # df_wgr = df_wgr[df_wgr['GR']>=10]
  seq = df_wgr.to_numpy() #create the series with gamma rays
  mseg = []
  dep_list = []
  for i in range(len(seq) - wsize + 1):
      dep = df_wgr.index[i]+int(wsize/2) #depth the sequence
      mseg.append(np.transpose(seq[i: i + wsize]))
      dep_list.append(dep)

  return np.array(mseg), dep_list


def plot_pred_distribution(td, pred_m, df_wm):

    fig, ax1 = plt.subplots(figsize = (10,5))

    ax1.set_xlabel('depth (ft)')
    ax1.set_ylabel('Probability')
    #plt.plot(df_gr[5500:]['GR'], color = 'black')
    ax1.fill_between(df_wm[5500:]['Depth'], df_wm[5500:]['Marcel']*100,  color = 'lightcoral', alpha = 0.5, label= 'prob_M')
    ax1.fill_between(df_wm[5500:]['Depth'], df_wm[5500:]['Sylvain']*100, color = 'lightblue', alpha = 0.5, label= 'prob_S')
    ax1.fill_between(df_wm[5500:]['Depth'], df_wm[5500:]['Conrad']*100, color = 'lightgreen', alpha = 0.5, label= 'prob_C')
    #ax2.legend()

    ax2 = ax1.twinx()
    ax2.set_ylabel('GR')
    ax2.plot(df_wm[5500:]['Depth'], df_wm[5500:]['GR'], color = 'black')
    ax2.axvline(td[0], color='red', label = 'actual_M')
    ax2.axvline(td[1], color='blue', label = 'actual_S')
    ax2.axvline(td[2], color='green', label = 'actual_C')

    ax2.axvline(pred_m[0], color='red', linestyle = '--', label= 'pred_M')
    ax2.axvline(pred_m[1], color='blue',linestyle = '--', label= 'pred_S')
    ax2.axvline(pred_m[2], color='green', linestyle = '--', label= 'pred_C')
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=6)

    plt.show()


def apply_label(wellName,d,df_log,df_tops):
    depths = df_tops[df_tops.index==wellName].values[0]
    if d < depths[0]:
        return 'Unknown'
    elif d > depths[0] and d<depths[1]:
        return 'Marcel'
    elif d > depths[1] and d<depths[2]:
        return 'Sylvain'
    elif d > depths[2] :
        return 'Conrad'


def plot_simple(wellName,df_log,df_tops):
    """
    Plot original time series data for GR and Depth and the marker positions for a chosen well.
    """

    sns.set(style="whitegrid")

    # Data filtering and processing
    filtered_df = df_log[(df_log['wellName'] == wellName)]
    depths = df_tops[df_tops.index == wellName].values[0]
    filtered_df = filtered_df[filtered_df['DEPTH'] > depths[0] - 200]
    filtered_df.loc[filtered_df['GR'] == -1, 'GR'] = filtered_df['GR'].mean()
    filtered_df['label'] = filtered_df['DEPTH'].apply(lambda x: apply_label(wellName, x,df_log,df_tops))

    # Define color map for labels
    palette = sns.color_palette("gist_earth",4)
    

    # Plotting seaborn lineplot 
    plt.figure(figsize=(8, 3))
    sns.lineplot(data=filtered_df, x='DEPTH', y='GR', hue='label', palette=palette, legend='full', alpha=0.5,linewidth=1)


    # Plotting vertical line for starting depth
    previous_depth = 0
    offset_increment = 20

    for depth, color in zip(depths, palette):
        plt.axvline(x=depth, color=color, linestyle='-', linewidth=1,alpha=1)
        offset = 0 if depth - previous_depth > 50 else offset_increment  # Adjust conditions based on your data scale
        y_position = plt.gca().get_ylim()[0] + offset
        plt.text(depth + 5, y_position, f'{depth}', color=color, verticalalignment='bottom',fontsize=8,alpha=1)
        previous_depth = depth

    plt.title(f'Gamma Ray (GR) vs Depth for Well {wellName}')
    plt.xlabel('Depth')
    plt.ylabel('Gamma Ray (GR)')
    plt.legend(title='Marker', fontsize=8)
    plt.show()


def get_classifier(X_train_transformed,y_train,X_test_transformed,y_test):
    """Fit XGBoost and Logistic Classifier"""
    eps = 1e-6
    f_mean = X_train_transformed.mean(axis=0)
    f_std = X_train_transformed.std(axis=0) + eps
    X_train_norm = (X_train_transformed - f_mean) / f_std
    X_valid_norm = (X_test_transformed - f_mean) / f_std
    # print(X_train_norm.shape)

    ##XGBoost
    classifier_xgb = xgb.XGBClassifier(max_depth=5, n_estimators=100,n_jobs=-1)
    classifier_xgb.fit(X_train_norm, y_train)
    preds = classifier_xgb.predict(X_valid_norm)

    print("XGBoost",(preds == y_test).mean())

    ##Logistic
    C = 1e-2
    classifier = LogisticRegression(penalty='l2', C=C, n_jobs=-1)
    classifier.fit(X_train_norm, y_train)
    probas = classifier.predict_proba(X_train_norm)
    train_score = classifier.score(X_train_norm, y_train)
    val_score = classifier.score(X_valid_norm, y_test)
    print('Logistic','{:2} eps: {:.2E}  C: {:.2E}   train_acc: {:.5f}  valid_acc: {:.5f}'.format(1, eps, C, train_score, val_score))

    return f_mean, f_std,classifier_xgb, classifier

def get_markers_rocket_order(f_mean,f_std,df_test_log, well, pred_column, wsize, input_variable, rocket, classifier_xgb,classifier, xgb):
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
        classifier_xgb : XGBoost classifier for prediction
        classifier : Logistic classifier for prediction
        xgb [bool]: whether XGBoost is used.
   
    Returns
    -------
        pred_m [list]: Predicted depths for each marker 
        df_wm [DataFrame]: DataFrame containing predicted probabilities for all markers.
    """

    df_gr = dfwellgr(well, df_test_log, input_variable)
    well_seq, dep_seq = window(df_gr, wsize)
    try:
        test_dr_well = rocket.transform(well_seq)
    except Exception as e:
        well_seq = well_seq.reshape(well_seq.shape[0], -1)
        test_dr_well = rocket.transform(well_seq)

    # normalize 'per feature'
    test_well = (test_dr_well - f_mean) / f_std

    if xgb:
        test_prob = classifier_xgb.predict_proba(test_well)
    else:
        test_prob = classifier.predict_proba(test_well)


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


# EVALUATION FUNCTIONS
def recall_tops(df_tops_true, df_tops_pred, tolerance = 4):

    if set(df_tops_true.columns) == set(df_tops_pred.columns) :
        concat_df = df_tops_true.copy()
        for col in df_tops_pred.columns:
            concat_df[col+"_pred"] = df_tops_pred[col]
        tp = 0
        p = 0
        mae = 0
        for col in df_tops_true:
            diffname = "{0}_ae".format(col)
            tpname = "{0}_tp".format(col)
            p += concat_df[col].count()
            concat_df[diffname] = concat_df[col]-concat_df[str(col + "_pred")]
            concat_df[diffname] = concat_df[diffname].abs()
            concat_df[tpname] = concat_df[diffname] <= tolerance
            tp += concat_df[tpname].sum()
            mae += concat_df[diffname].sum()
        return tp/p, mae/p, concat_df
    else :
        print("the tops columns are not valid")
    return None,None,None


def find_optimal_tolerance(df_test_tops, df_tops_pred):
    tolerance = 1
    while True:
        recall, mae, _ = recall_tops(df_test_tops, df_tops_pred, tolerance)
        if recall == 1:
            return tolerance
        tolerance += 1

def apply_evaluate(df_test_tops,df_tops_pred):
        
        tr = [20, 15, 10, 5]
        for tolerance in tr:
            recall, mae, df_result = recall_tops(df_test_tops,df_tops_pred,tolerance)
            print("tolerance {0}, recall {1}, mae {2}".format(tolerance, recall, mae))

        optimal_tolerance = find_optimal_tolerance(df_test_tops, df_tops_pred)

        print(f"Largest Error MARCEL: {df_result['MARCEL_ae'].max()}")
        print(f"Largest Error SYLVAIN: {df_result['SYLVAIN_ae'].max()}")
        print(f"Largest Error CONRAD: {df_result['CONRAD_ae'].max()}")
        print("🍺 Optimal Tolerance :", optimal_tolerance)
        return optimal_tolerance,df_result
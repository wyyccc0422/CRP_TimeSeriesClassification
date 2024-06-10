import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import scipy as sp



    

def fit_gaussian_process(df_tops, colmun_name, confidence_level=0.96):
    """
    Fits a Gaussian process regression model for depth measurements and find the confidence intervals.

    Returns:
    -----------
       intervals [dict]: A dictionary containing confidence intervals for each marker.
    """
    
    y = df_tops[colmun_name].dropna().values
    X = np.atleast_2d(np.arange(len(y))).T	# creates a 2d column vectors of indices (based on len(y)) // each represent the position of each depth measurement in the array
    
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-4, 1e2))

    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-10, normalize_y=True)
    gp.fit(X, y)
    
    # Predict with confidence intervals
    y_pred, sigma = gp.predict(X, return_std=True)
    confidence_interval = sigma * np.sqrt(2) * sp.special.erfinv(confidence_level)
    
    lower_b = np.min(y_pred - confidence_interval)
    upper_b = np.max( y_pred + confidence_interval)

    return (lower_b, upper_b)

def fit_lognormal_process(df_tops,colmun_name, confidence_level=0.96):
    """
    Fits a Gaussian process regression model for depth measurements and find the confidence intervals.
    Using log-normal distribution.
   
    Returns:
    -----------
       intervals [dict]: A dictionary containing confidence intervals for each marker.
    """

    ## transform to log
    y = np.log(df_tops[colmun_name].dropna().values)
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


    return (ci_low_original,ci_high_original)


def apply_process(df_tops,confidence_level=0.96,log=False):

    if log:
        intervals = fit_lognormal_process(df_tops, confidence_level)
    else:
        intervals = fit_gaussian_process(df_tops, confidence_level)
    for k,(v1,v2) in intervals.items():
        print(k,round(v1,0),round(v2,0))
    return intervals



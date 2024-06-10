import numpy as np
import pandas as pd
from scipy.stats import levy_stable, cauchy 
from scipy.optimize import minimize
from scipy.fft import fft, ifft
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in subtract")



def fit_stable_distribution(df_tops, column_name, confidence_level=0.95):
    y = df_tops[column_name].dropna().values
    
    # Initial guesses for stable distribution parameters
    ### Alpha = Stability ; Beta = Skewness ; Gamma = Scale ; Delta = Location
    alpha, beta = 1.8, 0.0                  #common guesses
    gamma, delta = np.std(y), np.mean(y)    #init to std and mean of data
    
    # Define Neg of Log Likelihood to be minimized later 
    def neg_log_likelihood(params):
        alpha, beta, gamma, delta = params
        #we use the levy_stable distribution from scipy
        #work with the log(PDF) --> higher numerical stability (floating point underflow in low probas)
        return -np.sum(levy_stable.logpdf(y, alpha, beta, loc=delta, scale=gamma))
    
    # Bounds : alpha [0.1, 2] , beta [-1, 1], gamma [0.0001, None], delta [None, None]
    # None : means can take any real value
    bounds = [(0.1, 2), (-1, 1), (0.0001, None), (None, None)]
    
    # Min neg_log_likelihood
    result = minimize(neg_log_likelihood, (alpha, beta, gamma, delta), bounds=bounds)
    alpha, beta, gamma, delta = result.x
    
    # Fit levy stable with the params
    dist = levy_stable(alpha, beta, loc=delta, scale=gamma)
    
    # Calculate confidence intervals
    lower_p = (1 - confidence_level) / 2
    upper_p = 1 - lower_p
    lower_b = dist.ppf(lower_p)
    upper_b = dist.ppf(upper_p)
    
    return (lower_b, upper_b)


###2nd way : Inverse Fast Fourrier Transform 

def fit_stable_distribution_fft(df_tops, column_name, confidence_level=0.95):
    y = df_tops[column_name].dropna().values
    n = len(y)
    
    # Empirical characteristic function (ECF)
    t = np.linspace(-np.pi, np.pi, n)
    ecf = fft(y * np.exp(1j * np.outer(t, y))).mean(axis=1)
    
    # Function to compute theoretical CF of a stable distribution
    def stable_cf(t, alpha, beta, gamma, delta):
        if alpha != 1:
            return np.exp(-gamma**alpha * np.abs(t)**alpha * (1 - 1j * beta * np.sign(t) * np.tan(np.pi * alpha / 2)) + 1j * delta * t)
        else:
            return np.exp(-gamma * np.abs(t) * (1 + 1j * beta * np.sign(t) * 2 / np.pi * np.log(np.abs(t))) + 1j * delta * t)

    # Minimize difference between ECF & CF
    def cf_difference(params):
        alpha, beta, gamma, delta = params
        cf_theoretical = stable_cf(t, alpha, beta, gamma, delta)
        return np.sum(np.abs(ecf - cf_theoretical)**2)

    # Initial guesses
    alpha, beta, gamma, delta = 1.8, 0, np.std(y), np.mean(y)
    bounds = [(0.1, 2), (-1, 1), (0.0001, None), (None, None)]

    # Minimize the CF difference
    result = minimize(cf_difference, (alpha, beta, gamma, delta), bounds=bounds)
    alpha, beta, gamma, delta = result.x


    # Placeholder for quantile computation // can adjust to the parameters and discretionary choices
    lower_b, upper_b = delta - 2*gamma, delta + 2*gamma  

    return (lower_b, upper_b)



###Yet another method : Cauchy Stable Distribution
def fit_cauchy_distribution(df_tops, column_name, confidence_level=0.95):
    y = df_tops[column_name].dropna().values
    
    # Initial guesses // recall : no mean nor variance for Cauchy distrib 
    location, scale = np.median(y), np.std(y) / 2  # median is more robust estimator for location
    
    # Define the negative log likelihood to be minimized
    def neg_log_likelihood(params):
        location, scale = params
        # Using the log(PDF) of the Cauchy distribution
        return -np.sum(cauchy.logpdf(y, loc=location, scale=scale))
    
    # Bounds for the params: location --> any real number, scale --> must be positive& not zero (DivisionByZero Error in PDF)
    bounds = [(None, None), (1e-4, None)]  
    
    # Minimize negative log likelihood
    result = minimize(neg_log_likelihood, (location, scale), bounds=bounds)
    location, scale = result.x
    
    # Fit Cauchy distribution with the params estimated
    dist = cauchy(location, scale)
    
    # Calculate confidence intervals
    lower_p = (1 - confidence_level) / 2
    upper_p = 1 - lower_p
    lower_b = dist.ppf(lower_p)
    upper_b = dist.ppf(upper_p)
    
    return (lower_b, upper_b)


# Cauchy Stable Distribution FFT

def fit_cauchy_distribution_fft(df_tops, column_name, confidence_level=0.95):
    y = df_tops[column_name].dropna().values
    
    # FFT of the empirical data
    N = len(y)
    data_fft = fft(y)
    frequencies = np.fft.fftfreq(N)
    
    # Initial guesses: using median and interquartile range for robustness
    location_guess = np.median(y)
    scale_guess = np.subtract(*np.percentile(y, [75, 25])) / 2
    
    def cauchy_cf(t, location, scale):
        return np.exp(1j * t * location - scale * np.abs(t))
    
    def objective_function(params):
        location, scale = params
        cauchy_fft = cauchy_cf(frequencies, location, scale)
        diff = np.abs(data_fft - cauchy_fft)
        return np.sum(diff)
    
    # Bounds for the parameters: location (any real number), scale (positive and not zero)
    bounds = [(None, None), (1e-4, None)]
    
    # Minimize the objective function
    result = minimize(objective_function, [location_guess, scale_guess], bounds=bounds)
    location, scale = result.x
    
    # Fit Cauchy distribution with the estimated parameters
    dist = cauchy(location, scale)
    
    # Calculate confidence intervals
    lower_p = (1 - confidence_level) / 2
    upper_p = 1 - lower_p
    lower_b = dist.ppf(lower_p)
    upper_b = dist.ppf(upper_p)

    return (lower_b, upper_b)
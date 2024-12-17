import numpy as np

def nash_sutcliffe_efficiency(actual, predicted):
    """
    Calculate Nash-Sutcliffe Efficiency (NSE)
    
    Parameters:
    actual : array-like
        actual data
    predicted : array-like
        predicted data
    
    Returns:
    float
        NSE value
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    numerator = np.sum((actual - predicted) ** 2)
    denominator = np.sum((actual - np.mean(actual)) ** 2)
    
    nse = 1 - (numerator / denominator)
    
    return nse

def root_mean_square_error(actual, predicted):
    """
    Calculate Root-Mean-Square Error (RMSE)
    
    Parameters:
    actual : array-like, actual data
    predicted : array-like, predicted data
    
    Returns: float, RMSE value
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    
    return rmse

def normalized_root_mean_square_error(actual, predicted, normalization='range'):
    """
    Calculate Normalized Root-Mean-Square Error (NRMSE)
    
    Parameters:
    actual : array-like, actual data
    predicted : array-like, predicted data
    normalization : str, Normalization method ('range', 'mean', 'std')
    
    Returns: float, NRMSE value
    """
    rmse = root_mean_square_error(actual, predicted)
    
    if normalization == 'range':
        norm_factor = np.max(actual) - np.min(actual)
    elif normalization == 'mean':
        norm_factor = np.mean(actual)
    elif normalization == 'std':
        norm_factor = np.std(actual)
    else:
        raise ValueError("Normalization method must be 'range', 'mean', or 'std'")
    
    nrmse = rmse / norm_factor
    
    return nrmse

def percent_bias(actual, predicted):
    """
    Calculate Percent Bias (PBIAS)
    
    Parameters:
    actual : array-like, actual data
    predicted : array-like, predicted data
    
    Returns: float, PBIAS value
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    pbias = 100 * (np.sum(predicted - actual) / np.sum(actual))
    
    return pbias

def calculate_r(Y_actual, Y_predicted):
    """
    Calculate the coefficient of determination (R^2) for the given actual and predicted values.

    Parameters:
    Y_actual (array-like): Actual values.
    Y_predicted (array-like): Predicted values.

    Returns:
    float: The coefficient of determination (R^2).
    """
    # Convert inputs to numpy arrays
    Y_actual = np.array(Y_actual)
    Y_predicted = np.array(Y_predicted)
    
    return np.corrcoef(Y_actual,Y_predicted)[0, 1]

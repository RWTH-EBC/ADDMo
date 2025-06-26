"""
by Jun Jiang
"""
import numpy as np
from typing import Union


def calculate_data_distribution(data: Union[list, np.ndarray]):
    """
    Calculate the data distribution (e.g. for a box plot analysis)
    :param data: Data of a list or a ndarray
    :return: Dict of results
    """
    data = np.array(data)
    res_dict = {
        'N': len(data),  # Amount of sample
        'median': np.median(data),  # Median
        'mean': np.mean(data),  # Mean value
        'q0': np.min(data),  # Min
        'q2.5': np.percentile(data, 2.5),  # 2.5% percentile
        'q5': np.percentile(data, 5),  # 5% percentile
        'q25': np.percentile(data, 25),  # 25% percentile
        'q50': np.percentile(data, 50),  # 50% percentile
        'q75': np.percentile(data, 75),  # 75% percentile
        'q95': np.percentile(data, 95),  # 95% percentile
        'q97.5': np.percentile(data, 97.5),  # 97.5% percentile
        'q100': np.max(data)  # Max
    }
    iqr = res_dict['q75'] - res_dict['q25']
    iqr_lower_bound = res_dict['q25'] - 1.5 * iqr
    iqr_upper_bound = res_dict['q75'] + 1.5 * iqr
    iqr_outliers = [x for x in data if x < iqr_lower_bound or x > iqr_upper_bound]
    res_dict.update({
        'N_IQR_outliers': len(iqr_outliers),  # Samples of IQR (inter-quartile range) outliers
        'ratio_IQR_outliers': len(iqr_outliers) / res_dict['N']  # Ratio of IQR-outliers
    })
    return res_dict


def calculate_mae(y_act: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the mean absolute error (mae)
    :param y_act: Actual values or true value
    :param y_pred: Predicted values
    :return: The mean-absolute-error (mae) between y_act and y_pred
    """
    abs_e = np.abs(y_act - y_pred)  # Abs. errors
    mae = np.mean(abs_e)
    return mae


def calculate_rmse(y_act: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the root-mean-squared-error (rmse) between actual values and predicted values
    :param y_act: Actual values or true value
    :param y_pred: Predicted values
    :return: The root-mean-squared-error (rmse) between y_act and y_pred
    """
    se = (y_act - y_pred) ** 2  # Squared errors
    mse = np.mean(se)  # Mean squared errors
    rmse = np.sqrt(mse)
    return rmse


def calculate_r_squared(y_act: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the coefficient of determination (R-squared) between actual values and predicted values
    :param y_act: Actual values or true value
    :param y_pred: Predicted values
    :return: The coefficient of determination (R-squared) between y_act and y_pred
    """
    y_mean = np.mean(y_act)
    RSS = np.sum((y_act - y_pred) ** 2)  # Residual sum of squares
    TSS = np.sum((y_act - y_mean) ** 2)  # Total sum of squares
    r_squared = 1 - RSS / TSS
    return r_squared


if __name__ == '__main__':
    random_numbers = np.random.rand(100000)*100
    print(calculate_data_distribution(data=random_numbers))

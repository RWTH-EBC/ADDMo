import numpy as np
from numpy import ndarray


def rearrange_training_data(x_train: ndarray, x_val: ndarray, y_train: ndarray, y_val: ndarray) -> \
        tuple[ndarray, ndarray, ndarray, ndarray]:
    """Moves outliers from training data to validation data

    Parameters
    ----------
    x_train: ndarray
        NxD matrix of training data, N: number of data points, D: number of dimensions
    x_val: ndarray
         MxD matrix of validation data, M: number of data points, D: number of dimensions
    y_train: ndarray
        Nx1 matrix, N: number of data points; Classification: 0: Normal data, 1: Outlier;
    y_val: ndarray
        Mx1 matrix, M: number of data points; Classification: 0: Normal data, 1: Outlier;

    Returns
    -------
     tuple[ndarray, ndarray, ndarray, ndarray]
        x_train, x_val, y_train, y_val
    """
    x_train_in = x_train[y_train == 0, :]
    x_train_out = x_train[y_train == 1, :]
    x_val = np.concatenate((x_train_out, x_val))

    y_train_in = y_train[y_train == 0]
    y_train_out = y_train[y_train == 1]
    y_val = np.concatenate((y_train_out, y_val))

    return x_train_in, x_val, y_train_in, y_val

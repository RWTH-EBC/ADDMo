import os
import pickle
from typing import Any

import pandas as pd
import numpy as np
import json
from pandas import DataFrame



def split_simulation_data(
    df: DataFrame,
    train_val_test_indices: list,
    val_fraction: float,
    test_fraction: float,
    shuffle: bool = False,
) -> dict:
    """Splits data in training, validation, test and remaining splits; last column of dataframe will be used as y

    Parameters
    ----------
    df: DataFrame
        data to be process
    train_val_test_indices: list
        list of indices to be used as training, validation and test split
    val_fraction: float
        relative fraction of data points from dataIndices to be used as validation split
    test_fraction: float
            relative fraction of data points from dataIndices to be used as test split
    random_state: int
        random state for shuffling
    shuffle: bool
        If set to True, training, validation and test datapoints will be shuffled

    Returns
    -------
    dict
        {'available_data': available_data, 'x_remaining', 'y_remaining', 'header'
    """

    # select list of indices used as training, validation and test split
    data = df.iloc[train_val_test_indices]

    # Shuffle data
    if shuffle:
        data_shuffled = data.sample(frac=1, random_state=1)
    else:
        data_shuffled = data




    # training split
    training_split = 1 - val_fraction - test_fraction
    xy_training = data_shuffled.iloc[0 : round(len(data_shuffled) * training_split)]

    # validation split
    xy_validation = data_shuffled.iloc[
        round(len(data_shuffled) * training_split): round(
            len(data_shuffled) * (training_split + val_fraction)
        )
    ]

    # test split
    xy_test = data_shuffled.iloc[
              round(len(data_shuffled) * (training_split + val_fraction)):
    ]

    # remaining split (not train/val/test)
    xy_remaining = df.drop(train_val_test_indices)

    return xy_training, xy_validation, xy_test, xy_remaining

def move_true_invalid_from_training_2_validation(xy_train, xy_val, true_validity_train, true_validity_val):
    """Moves true invalid from training data to validation data"""

    # Convert 0s and 1s to False and True first for easier handling
    true_validity_train = true_validity_train.astype(bool)
    true_validity_val = true_validity_val.astype(bool)

    # Select the "true invalid" data points from xy_train
    invalid_data = xy_train[~true_validity_train]
    invalid_labels = true_validity_train[~true_validity_train]

    # Remove the "true invalid" data points from xy_train
    xy_train_new = xy_train[true_validity_train]
    true_validity_train_new = true_validity_train[true_validity_train]

    # Append the "true invalid" data points to xy_val
    xy_val_new = pd.concat([xy_val, invalid_data])
    true_validity_val_new = pd.concat([true_validity_val, invalid_labels])


    return xy_train_new, xy_val_new, true_validity_train_new, true_validity_val_new


def write_pkl(data, filename: str, directory: str = None, override: bool = False):
    """Writes data to a pickle file.

    Parameters
    ----------
    data:
        The object that is supposed to be saved.
        The name of the file.
    directory: str
        The directory the file will be saved to.
    override: Boolean
        If true, existing data will be overwritten
    """

    path = _get_path(filename, directory)

    # make sure the directory does exist
    if directory is not None:
        if not os.path.exists(directory):
            os.mkdir(directory)

    # make sure the file does not already exist
    if os.path.exists(path) and not override:
        if not _get_bool(
            f'The file "{path}" already exists. Do you want to override it?\n'
        ):
            return 0

    # open the path and write the file
    pkl_file = open(path, "wb")
    pickle.dump(data, pkl_file)

    # close file
    pkl_file.close()

def read_pkl(filename: str, directory: str = None) -> Any:
    """Reads data from a pickle file.

    Parameters
    ----------
    filename: str
        The name of the file.
    directory: str
        The directory the file is located.

    Returns
    -------
    object
        Pickle object.
    """

    path = _get_path(filename, directory)

    if os.path.exists(path):  # check for the existence of the path
        pkl_file = open(path, "rb")  # open path
        pkl_data = pickle.load(pkl_file)  # read data
        pkl_file.close()  # close path

        if pkl_data is not None:
            return pkl_data  # return data
        else:
            raise FileNotFoundError(f"No data at {path} found.")
    else:
        raise FileNotFoundError(f"The path {path} does not exist.")

def _get_path(filename: str, directory: str) -> str:
    """
    Returns the full path for a given filename and directory.
    """
    if directory is not None:  # check if directory is none
        if not os.path.exists(directory):  # check if path exists
            os.makedirs(directory)  # create new directory
        return str(directory + "\\" + filename)  # calculate full path
    else:
        return filename


def _get_bool(message: str, true: list = None, false: list = None) -> bool or str:
    if false is None:
        false = ["no", "nein", "false", "1", "n"]
    if true is None:
        true = ["yes", "ja", "true", "wahr", "0", "y"]

    val = input(message).lower()
    if val in true:
        return True
    elif val in false:
        return False
    else:
        print("Please try again.")
        print("True:", true)
        print("False:", false)
        _get_bool(message, true, false)

    return val

def write_csv(data: pd.DataFrame, filename: str, directory: str = None, overwrite: bool = True):
    """Writes data to a CSV file.

    Parameters
    ----------
    data: pd.DataFrame
        The DataFrame that is supposed to be saved.
    filename: str
        The name of the file.
    directory: str
        The directory the file will be saved to.
    overwrite: Boolean
        If true, existing data will be overwritten
    """

    filename = filename + ".csv"

    path = _get_path(filename, directory)

    # make sure the directory does exist
    if directory is not None:
        if not os.path.exists(directory):
            os.mkdir(directory)

    # make sure the file does not already exist
    if os.path.exists(path) and not overwrite:
        if not _get_bool(
            f'The file "{path}" already exists. Do you want to override it?\n'
        ):
            return 0

    # Write DataFrame to CSV
    data.to_csv(path, sep=";", index=True, header=True, encoding="utf-8")

def read_csv(filename: str, directory: str = None, **kwargs) -> pd.DataFrame:
    """Reads data from a CSV file.

    Parameters
    ----------
    filename: str
        The name of the file.
    directory: str
        The directory the file is located.

    Returns
    -------
    pd.DataFrame
        DataFrame object.
    """

    if "index_col" in kwargs:
        index_col = kwargs["index_col"]
    else:
        index_col = 0

    filename = filename + ".csv"

    path = _get_path(filename, directory)

    if os.path.exists(path):  # check for the existence of the path
        return pd.read_csv(path, sep=";", dtype="float", encoding="unicode_escape", index_col=index_col)
    else:
        raise FileNotFoundError(f"The path {path} does not exist.")


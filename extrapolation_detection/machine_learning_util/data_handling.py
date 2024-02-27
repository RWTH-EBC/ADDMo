import os
import pickle
from typing import Any

import pandas as pd
import numpy as np
from pandas import DataFrame

def load_csv(filename: str, path: str = "data") -> DataFrame:
    """Loads csv file as dataframe"""
    if not filename.endswith(".csv"):
        filename = filename + ".csv"
    if len(path) > 0:
        file = path + "/" + filename
    else:
        file = filename
    return pd.read_csv(file, delimiter=";", dtype="float", encoding="unicode_escape")


def split_simulation_data(
    df: DataFrame,
    train_val_test_indices: list,
    val_fraction: float,
    test_fraction: float,
    random_state: int = 0,
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
        data_shuffled = data.sample(frac=1, random_state=random_state)
    else:
        data_shuffled = data
    training_split = 1 - val_fraction - test_fraction

    # training data
    xy_training = data_shuffled.iloc[0 : round(len(data_shuffled) * training_split)]
    # validation data
    xy_validation = data_shuffled.iloc[
        round(len(data_shuffled) * training_split) : round(
            len(data_shuffled) * (training_split + val_fraction)
        )
    ]
    # test data
    xy_test = data_shuffled.iloc[
              round(len(data_shuffled) * (training_split + val_fraction)):
    ]

    # Last column will be used as y, the rest of the dataframe will be used as x
    x_val = xy_validation.iloc[:, :-1].to_numpy()
    y_val = xy_validation.iloc[:, -1].to_numpy().reshape((-1, 1))
    x_test = xy_test.iloc[:, :-1].to_numpy()
    y_test = xy_test.iloc[:, -1].to_numpy().reshape((-1, 1))
    x_train = xy_training.iloc[:, :-1].to_numpy()
    y_train = xy_training.iloc[:, -1].to_numpy().reshape((-1, 1))

    # Select remaining data not used for training, validation and testing
    data = df.drop(train_val_test_indices)
    x_remaining = data.iloc[:, :-1].to_numpy()
    y_remaining = data.iloc[:, -1].to_numpy().reshape((-1, 1))



    xy_tot_splitted = {
        "available_data": {
            "x_train": x_train,
            "y_train": y_train,
            "x_val": x_val,
            "y_val": y_val,
            "x_test": x_test,
            "y_test": y_test,
        },
        "non_available_data": {  # remaining not in training, validation and test (full year simulation)
            "x_remaining": x_remaining,
            "y_remaining": y_remaining,
        },
        "header": np.array(df.columns),
    }

    return xy_tot_splitted


def write_pkl(data, filename: str, directory: str = None, override: bool = False):
    """Writes data to a pickle file.

    Parameters
    ----------
    data:
        The object that is supposed to be saved.
    filename: str
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


def _get_path(filename: str, directory: str) -> str:
    """
    Returns the full path for a given filename and directory.
    """
    if ".pkl" not in filename:  # check for file extension
        filename += ".pkl"  # add file extension

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

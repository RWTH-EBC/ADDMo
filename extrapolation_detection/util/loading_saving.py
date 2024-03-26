import os
import pickle
from typing import Any

import pandas as pd


def write_pkl(data, filename: str, directory: str = None, override: bool = True):
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

    filename = filename + ".pkl"

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

    filename = filename + ".pkl"

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
        return pd.read_csv(path, sep=";", dtype="float", encoding="utf-8", index_col=index_col)
    else:
        raise FileNotFoundError(f"The path {path} does not exist.")

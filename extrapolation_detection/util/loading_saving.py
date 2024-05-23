import os
import pickle
from typing import Any
import pandas as pd
import glob
from core.util.load_save_utils import create_path_or_ask_to_override, get_path


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

    path = create_path_or_ask_to_override(filename, directory, override)

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

    path = get_path(filename, directory)

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

    path = create_path_or_ask_to_override(filename, directory, overwrite)

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

    path = get_path(filename, directory)

    if os.path.exists(path):  # check for the existence of the path
        return pd.read_csv(path, sep=";", dtype="float", encoding="utf-8", index_col=index_col)
    else:
        raise FileNotFoundError(f"The path {path} does not exist.")

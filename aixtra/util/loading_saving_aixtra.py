import os
import pickle
from typing import Any
import pandas as pd
import glob
from addmo.util.load_save_utils import create_path_or_ask_to_override, get_path
from addmo.s3_model_tuning.models.model_factory import ModelFactory


def write_pkl(data, filename: str, directory: str = None, override: bool = True):
    """Writes system_data to a pickle file.

    Parameters
    ----------
    data:
        The object that is supposed to be saved.
        The name of the file.
    directory: str
        The directory the file will be saved to.
    override: Boolean
        If true, existing system_data will be overwritten
    """

    filename = filename + ".pkl"

    path = create_path_or_ask_to_override(filename, directory, override)

    # open the path and write the file
    pkl_file = open(path, "wb")
    pickle.dump(data, pkl_file)

    # close file
    pkl_file.close()


def read_pkl(filename: str, directory: str = None) -> Any:
    """Reads system_data from a pickle file.

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
        pkl_data = pickle.load(pkl_file)  # read system_data
        pkl_file.close()  # close path

        if pkl_data is not None:
            return pkl_data  # return system_data
        else:
            raise FileNotFoundError(f"No system_data at {path} found.")
    else:
        raise FileNotFoundError(f"The path {path} does not exist.")


def write_csv(data: pd.DataFrame, filename: str, directory: str = None, overwrite: bool = True):
    """Writes system_data to a CSV file.

    Parameters
    ----------
    data: pd.DataFrame
        The DataFrame that is supposed to be saved.
    filename: str
        The name of the file.
    directory: str
        The directory the file will be saved to.
    overwrite: Boolean
        If true, existing system_data will be overwritten
    """

    filename = filename + ".csv"

    path = create_path_or_ask_to_override(filename, directory, overwrite)

    # Write DataFrame to CSV
    data.to_csv(path, sep=";", index=True, header=True, encoding="utf-8")


def read_csv(filename: str, directory: str = None, **kwargs) -> pd.DataFrame:
    """Reads system_data from a CSV file.

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


def load_regressor(filename, directory):
    """Loads a regressor model from a file, automatically determining the file type."""
    file_types = ['h5', 'joblib', 'onnx', 'keras']
    files_found = []

    # Find complete filepath
    for file_type in file_types:
        path_pattern = os.path.join(directory, f"{filename}.{file_type}")
        files_found.extend(glob.glob(path_pattern))

    if not files_found:
        raise FileNotFoundError(f"No model file found for {filename} in {directory} with supported types {file_types}")

    loaded_model = ModelFactory().load_model(files_found[0])
    return loaded_model

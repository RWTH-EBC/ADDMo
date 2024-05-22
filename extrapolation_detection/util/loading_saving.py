import os
import pickle
from typing import Any
import pandas as pd
from core.s3_model_tuning.models.keras_models import SciKerasSequential
import glob
from core.s3_model_tuning.models.model_factory import ModelFactory


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
        return os.path.join(directory, filename)  # calculate full path
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


def write_regressor(regressor, directory, filename, file_type=None):
    """Writes a regressor to a file based on its class type."""
    if isinstance(regressor, SciKerasSequential):  # Save as .h5 if regressor is Keras
        file_type = 'h5'
        regressor.save_regressor(directory=directory, filename=filename, file_type=file_type)
    else:
        if file_type is None:
            file_type = 'joblib'
        regressor.save_regressor(directory=directory, filename=filename, file_type=file_type)


def read_regressor(filename, directory):
    """Reads a regressor model from a file, automatically determining the file type."""
    file_types = ['h5', 'joblib', 'onnx']
    files_found = []

    for file_type in file_types:
        path_pattern = os.path.join(directory, f"{filename}.{file_type}")
        files_found.extend(glob.glob(path_pattern))

    if not files_found:
        raise FileNotFoundError(f"No model file found for {filename} in {directory} with supported types {file_types}")

    latest_file = max(files_found, key=os.path.getmtime)  # Use the last saved file for loading the regressor
    loaded_model = ModelFactory().load_model(latest_file)
    return loaded_model

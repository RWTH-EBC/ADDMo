import os
import shutil
import pandas as pd
import json
from pathlib import Path
from pydantic import FilePath, BaseModel
from typing import Type, TypeVar, Union
import glob
from core.s3_model_tuning.models.model_factory import ModelFactory

ConfigT = TypeVar("ConfigT", bound=BaseModel)


def load_config_from_json(
        config: Union[ConfigT, FilePath, str, dict], config_type: Type[ConfigT]
) -> ConfigT:
    """Generic config loader, either accepting a path to a json file, a json string, a
    dict or passing through a valid config object."""

    if isinstance(config, (str, Path)):
        # if we have a str / path, we need to check whether it is a file or a json string
        if Path(config).is_file():
            # if we have a valid file pointer, we load it
            with open(config, "r") as f:
                config = json.load(f)
        else:
            # since the str is not a file path, we assume it is json and try to load it
            try:
                config = json.loads(config)
            except json.JSONDecodeError as e:
                # if we failed, we raise an error notifying the user of possibilities
                raise TypeError(
                    f"The config '{config:.100}' is neither an existing file path, nor a "
                    f"valid json document."
                ) from e
    return config_type.model_validate(config)


def save_config_to_json(config: ConfigT, path: str):
    """Save the config to a json file."""
    config_json = config.model_dump_json(indent=4)
    with open(path, "w") as f:
        f.write(config_json)


def load_data(abs_path: str) -> pd.DataFrame:
    if abs_path.endswith(".csv"):
        # Read the CSV file
        df = pd.read_csv(
            abs_path, delimiter=";", index_col=[0], encoding="latin1", header=[0]
        )
    elif abs_path.endswith(".xlsx"):
        # Read the Excel file
        df = pd.read_excel(abs_path, index_col=[0], header=[0])

    # Convert the index to datetime
    df.index = pd.to_datetime(df.index, format="%d.%m.%Y %H:%M %Z")

    return df


def write_data(df: pd.DataFrame, abs_path: str):
    if abs_path.endswith(".csv"):
        # Write the CSV file
        df.to_csv(abs_path, sep=";", encoding="latin1")
    elif abs_path.endswith(".xlsx"):
        # Write the Excel file
        df.to_excel(abs_path)


def create_or_clean_directory(path: str) -> str: # Todo : move to load_save_utils
    if not os.path.exists(path):
        # Path does not exist, create it
        os.makedirs(path)
    elif not os.listdir(path):
        # Path exists, but is empty
        pass
    else:
        # Path exists, ask for confirmation to delete current contents
        response = input(f"The directory {path} already exists. To overwrite "
                         "the content type <y>, for deleting the current contents type <d>")
        if response.lower() == 'd':
            # Delete the contents of the directory
            for filename in os.listdir(path):
                file_path = os.path.join(path, filename)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
        elif response.lower() == 'y':
            pass
        else:
            print("Operation cancelled.")
            return None

    return path


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

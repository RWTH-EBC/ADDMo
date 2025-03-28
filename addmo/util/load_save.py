import pandas as pd
import json
from pathlib import Path
from pydantic import FilePath, BaseModel
from typing import Type, TypeVar, Union

ConfigT = TypeVar("ConfigT", bound=BaseModel)


def load_config_from_json(
        config: Union[ConfigT, FilePath, str, dict], config_type: Type[ConfigT]
) -> ConfigT:
    """
    Generic config loader, either accepting a path to a json file, a json string, a
    dict or passing through a valid config object.
    """

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
    """
    Save the config to a json file.
    """
    config_json = config.model_dump_json(indent=4)
    with open(path, "w") as f:
        f.write(config_json)


def load_data(abs_path: str) -> pd.DataFrame:
    """
    Load data from absolute file path.
    """

    if abs_path.endswith(".csv"):
        # Read the CSV file
        df = pd.read_csv(
            abs_path, delimiter=",", index_col=[0], encoding="latin1", header=[0]
        )
    elif abs_path.endswith(".xlsx"):
        # Read the Excel file
        df = pd.read_excel(abs_path, index_col=[0], header=[0])

    # Convert the index to datetime
    df.index = pd.to_datetime(df.index, format="%Y-%m-%d %H:%M:%S")

    return df


def write_data(df: pd.DataFrame, abs_path: str):
    """
    Write data to absolute file path.
    """
    if abs_path.endswith(".csv"):
        # Write the CSV file
        df.to_csv(abs_path, sep=";", encoding="latin1")
    elif abs_path.endswith(".xlsx"):
        # Write the Excel file
        df.to_excel(abs_path)



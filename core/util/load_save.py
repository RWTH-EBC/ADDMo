import os
import shutil

import pandas as pd
import yaml
def load_yaml_to_dict(path_to_yaml):
    """
    Load a yaml file to a dictionary.
    """
    with open(path_to_yaml, "r") as f:
        config = yaml.safe_load(f)
    return config

def save_yaml_from_dict(path_to_yaml, dict_to_save):
    """
    Save a dictionary to a yaml file.
    """
    with open(path_to_yaml, "w") as f:
        yaml.safe_dump(dict_to_save, f, default_flow_style=False)

def load_data(abs_path: str) -> pd.DataFrame:

    if abs_path.endswith(".csv"):
        # Read the CSV file
        df = pd.read_csv(abs_path, delimiter=';', index_col=[0], encoding='latin1', header=[0])
    elif abs_path.endswith(".xlsx"):
        # Read the Excel file
        df = pd.read_excel(abs_path, index_col=[0], header=[0])

    # Convert the index to datetime
    df.index = pd.to_datetime(df.index, format='%d.%m.%Y %H:%M %Z')

    return df

def write_data(df: pd.DataFrame, abs_path: str):
    if abs_path.endswith(".csv"):
        # Write the CSV file
        df.to_csv(abs_path, sep=';', encoding='latin1')
    elif abs_path.endswith(".xlsx"):
        # Write the Excel file
        df.to_excel(abs_path)


def create_or_override_directory(path: str) -> str:
    if not os.path.exists(path):
        # Path does not exist, create it
        os.makedirs(path)
    elif not os.listdir(path):
        # Path exists, but is empty
        pass
    else:
        # Path exists, ask for confirmation to delete current contents
        # response = input(f"The directory {path} already exists. Do you want to delete the current "
        #                  f"contents? (y/n): ")
        if True: #response.lower() == 'y': # Todo: uncomment for production
            # Delete the contents of the directory
            for filename in os.listdir(path):
                file_path = os.path.join(path, filename)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
        else:
            print("Operation cancelled.")
            return None

    return path

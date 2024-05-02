import os
import json
import subprocess
from core.util.load_save import create_or_clean_directory
from core.s2_data_tuning.config.data_tuning_config import DataTuningFixedConfig
from core.s3_model_tuning.config.model_tuning_config import ModelTuningExperimentConfig

def root_dir():
    # Finds the root directory of the git repository
    return os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

def raw_data_path(path: str = None):
    if path is None:
        # Use the default path
        return os.path.join(root_dir(), 'raw_input_data', 'InputData.xlsx')
    elif os.path.isabs(path):
        # If the provided path is absolute, return it as is
        return path
    else:
        # If the provided path is relative, join it with the 'raw_input_data' directory
        return os.path.join(root_dir(), 'raw_input_data', path)

def results_dir():
    return os.path.join(root_dir(), 'results')
def results_dir_wandb():
    return os.path.join(results_dir(), 'wandb')

def results_dir_data_tuning(config: DataTuningFixedConfig):
    path = os.path.join(root_dir(), results_dir(), config.name_of_raw_data, config.name_of_data_tuning)
    return create_or_clean_directory(path)


def results_dir_model_tuning(config: ModelTuningExperimentConfig):
    path = os.path.join(root_dir(), results_dir(), config.name_of_raw_data,
                        config.name_of_data_tuning_experiment, config.name_of_model_tuning_experiment)
    return create_or_clean_directory(path)

def ed_use_case_dir():
    return os.path.join(root_dir(), 'extrapolation_detection', 'use_cases')

def results_dir_extrapolation_experiment(experiment_name: str):
    path = os.path.join(
        ed_use_case_dir(),
        "results",
        experiment_name
    )
    return create_or_clean_directory(path)

def get_commit_id():

    try:
        commit_id= subprocess.check_output(["git", "describe", "--always"]).strip().decode()
    except subprocess.CalledProcessError:
        commit_id = 'Unknown'
    return commit_id


def load_metadata(abs_path: str):

    # Load metadata from a JSON file associated with the specified absolute path.

    filename= os.path.splitext(abs_path)[0]
    metadata_path = f"{filename}_metadata.json"

    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            metadata = json.load(f)
        return metadata
    else:
        raise FileNotFoundError(
            f'The metadata file {metadata_path} does not exist. Try saving the model before loading it or specify the path where the model is saved.'
        )
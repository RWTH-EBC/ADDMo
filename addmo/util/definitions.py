import os
import json
from addmo.util.load_save_utils import create_or_clean_directory
from addmo.s2_data_tuning.config.data_tuning_config import DataTuningFixedConfig
from addmo.s3_model_tuning.config.model_tuning_config import ModelTuningExperimentConfig


def root_dir():
    # Finds the root directory of the git repository
    return os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


def raw_data_path(path: str = None):
    if path is None:
        # Use the default path
        return os.path.join(root_dir(), 'addmo_examples', 'raw_input_data', 'InputData.xlsx')
    elif os.path.isabs(path):
        # If the provided path is absolute, return it as is
        return path
    else:
        # If the provided path is relative, join it with the 'raw_input_data' directory
        return os.path.join(root_dir(), 'raw_input_data', path)


def results_dir():
    return os.path.join(root_dir(), 'addmo_examples', 'results')


def results_dir_wandb():
    return os.path.join(results_dir(), 'wandb')


def results_dir_data_tuning(config: DataTuningFixedConfig):
    path = os.path.join(root_dir(), results_dir(), config.name_of_raw_data, config.name_of_tuning)
    return create_or_clean_directory(path)


def results_dir_model_tuning(config: ModelTuningExperimentConfig):
    path = os.path.join(root_dir(), results_dir(), config.name_of_raw_data,
                        config.name_of_data_tuning_experiment, config.name_of_model_tuning_experiment)
    return create_or_clean_directory(path)


def ed_use_case_dir():
    return os.path.join(root_dir(), 'aixtra_use_case')

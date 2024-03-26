import os
import git

from core.util.load_save import create_or_clean_directory
from core.s2_data_tuning.config.data_tuning_config import DataTuningFixedConfig
from core.s3_model_tuning.config.model_tuning_config import ModelTuningExperimentConfig
from extrapolation_detection.use_cases.config.ed_experiment_config import ExtrapolationExperimentConfig

def root_dir():
    # Finds the root directory of the git repository
    return git.Repo('.', search_parent_directories=True).working_tree_dir

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
    path = os.path.join(root_dir(), results_dir(), config.name_of_raw_data, config.name_of_tuning)
    return create_or_clean_directory(path)


def results_dir_model_tuning(config: ModelTuningExperimentConfig):
    path = os.path.join(root_dir(), results_dir(), config.name_of_raw_data,
                        config.name_of_data_tuning_experiment, config.name_of_model_tuning_experiment)
    return create_or_clean_directory(path)

def results_dir_extrapolation_experiment(config: ExtrapolationExperimentConfig):
    path = os.path.join(
        root_dir(),
        "extrapolation_detection",
        "use_cases",
        config.experiment_folder
    )
    return create_or_clean_directory(path)
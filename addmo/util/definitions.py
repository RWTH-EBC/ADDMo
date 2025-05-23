import os
import glob
import json
from addmo.util.load_save_utils import create_or_clean_directory, root_dir
from addmo.s2_data_tuning.config.data_tuning_config import DataTuningFixedConfig
from addmo.s3_model_tuning.config.model_tuning_config import ModelTuningExperimentConfig
from addmo.s1_data_tuning_auto.config.data_tuning_auto_config import DataTuningAutoSetup


def raw_data_path(path: str = None):
    """
    Returns the path to the raw input data file, using a default if none is provided.
    """
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
    """
    Returns the path to the results directory.
    """
    return os.path.join(root_dir(), 'addmo_examples', 'results')


def results_dir_wandb():
    """
   Returns the path to the results directory for wandb logging.
   """
    return os.path.join(results_dir(), 'wandb')


def results_dir_data_tuning(config: DataTuningFixedConfig, user_input='y'):
    """
    Returns the path to the results directory for data tuning based on config.
    """
    path = os.path.join(root_dir(), results_dir(), config.name_of_raw_data, config.name_of_tuning)
    return create_or_clean_directory(path, user_input)


def results_dir_model_tuning(config: ModelTuningExperimentConfig,user_input='y', ):
    """
    Returns the path to the results directory for model tuning based on config.
    """
    path = os.path.join(root_dir(), results_dir(), config.name_of_raw_data,
                        config.name_of_data_tuning_experiment, config.name_of_model_tuning_experiment)
    return create_or_clean_directory(path, user_input)


def ed_use_case_dir():
    """
    Returns the path to the use case directory.
    """
    return os.path.join(root_dir(), 'aixtra_use_case')

def return_results_dir_model_tuning( name_of_raw_data='test_raw_data',name_of_data_tuning_experiment='test_data_tuning', name_of_model_tuning_experiment='test_model_tuning'):
    """
      Returns the path to the results directory for completed model tuning .
      """
    path = os.path.join(root_dir(),root_dir(), results_dir(), name_of_raw_data, name_of_data_tuning_experiment, name_of_model_tuning_experiment)
    return path

def return_best_model(dir):
    """
    Returns the path to the best model based on the directory path.
    """
    model_files = glob.glob(os.path.join(dir, "best_model.*"))

    if model_files:
        path_to_regressor = model_files[0]  # Load the first match
        return path_to_regressor
    else:
        raise FileNotFoundError("No 'best_model' file found in the directory.")

def results_dir_data_tuning_auto(name_of_raw_data='test_raw_data', name_of_data_tuning_experiment='data_tuning_experiment_auto'):
    """
    Returns the directory of tuned data based on config's name of raw_data.
    """
    if name_of_raw_data is None:
        config = DataTuningAutoSetup()
        name_of_raw_data = config.name_of_raw_data
    dir = os.path.join(root_dir(), results_dir(), name_of_raw_data, name_of_data_tuning_experiment)
    return dir

def results_dir_data_tuning_fixed(name_of_raw_data='test_raw_data'):
    """
    Returns the path to the folder in results directory of tuned data based on config.
    """
    if name_of_raw_data is None:
        config = DataTuningFixedConfig()
        name_of_raw_data = config.name_of_raw_data
    dir = os.path.join(root_dir(), results_dir(), name_of_raw_data, 'data_tuning_experiment_fixed')
    return dir


def results_model_streamlit_testing(name_tuning_exp):
    """
    Returns the path to the results directory for model tuning based on config.
    """
    path = os.path.join(root_dir(), results_dir(), 'model_streamlit_test', name_tuning_exp)
    return path




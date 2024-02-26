import os
import git
import shutil
from core.s2_data_tuning.config.data_tuning_config import DataTuningFixedConfig
from core.s3_model_tuning.config.model_tuning_config import ModelTuningSetup
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
def config_files_path():
    return os.path.join(root_dir(), 'config')

def results_dir_data_tuning(config: DataTuningFixedConfig):
    path = os.path.join(root_dir(), results_dir(), config.name_of_raw_data, config.name_of_tuning)
    return _create_or_override_directory(path)


def results_dir_model_tuning(config: ModelTuningSetup):
    path = os.path.join(root_dir(), results_dir(), config.name_of_raw_data,
                        config.name_of_data_tuning_experiment, config.name_of_model_tuning_experiment)
    return _create_or_override_directory(path)

def _create_or_override_directory(path: str) -> str:
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
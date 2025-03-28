# This is an example working file of how to recreate auto data tuning using a previously saved config file for data tuning
# Model tuning is done automatically in the model testing script (if True), there is no need to execute this file separately in that case
import os
import json
import pandas as pd
from addmo.util.definitions import results_dir_data_tuning, results_dir
from addmo.util.load_save_utils import root_dir
from addmo.util.experiment_logger import ExperimentLogger
from addmo.util.experiment_logger import LocalLogger
from addmo.util.experiment_logger import WandbLogger
from addmo.s1_data_tuning_auto.config.data_tuning_auto_config import DataTuningAutoSetup
from addmo.s1_data_tuning_auto.data_tuner_auto import DataTunerAuto

def recreate_tuning(name_of_raw_data, name_of_tuning, abs_path_to_data, name_new_config):
    """
    Execute data tuning auto on new data based on existing configuration.
    """
    # Load the previously saved tuning configuration
    path_to_saved_config = os.path.join(root_dir(), results_dir(), name_of_raw_data, name_of_tuning, "config.json")

    with open(path_to_saved_config, "r") as file:
        saved_config_data = json.load(file)

    # Load new dataset here
    saved_config_data["abs_path_to_data"] = abs_path_to_data
    saved_config_data["name_of_raw_data"] = name_new_config
    saved_config_data["name_of_tuning"] = "test_data_tuning"

    # Convert the dictionary back to the DataTuningAutoSetup object
    new_config = DataTuningAutoSetup(**saved_config_data)

    # Configure the logger
    LocalLogger.directory = results_dir_data_tuning(new_config)
    LocalLogger.active = True
    WandbLogger.project = "addmo-test_data_auto_tuning"
    WandbLogger.directory = results_dir_data_tuning(new_config)
    WandbLogger.active = False

    ExperimentLogger.start_experiment(config=new_config)

    # Create a new tuner instance with the same tuning parameters
    new_tuner = DataTunerAuto(config=new_config)

    # Apply the same tuning process to the new data
    tuned_x_new = new_tuner.tune_auto()
    y_new = new_tuner.y

    tuned_xy_new = pd.concat([y_new, tuned_x_new], axis=1, join="inner").bfill()

    # Log the tuned system data
    ExperimentLogger.log_artifact(tuned_xy_new, name="tuned_xy_new", art_type="system_data")

    print("Recreated data tuning for new dataset successfully.")


if __name__ == "__main__":

    # Path to existing config and tuned data
    name_of_raw_data = "test_raw_data"
    name_of_tuning = "data_tuning_experiment_auto"
    abs_path_to_data = os.path.join(root_dir(),'addmo_examples','raw_input_data','InputData.xlsx')
    name_new_config = "test_raw_data_recreated"
    recreate_tuning(name_of_raw_data, name_of_tuning, abs_path_to_data, name_new_config)

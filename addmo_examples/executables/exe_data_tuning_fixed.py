import os
import json
import pandas as pd
from addmo.util.definitions import results_dir_data_tuning
from addmo.util.load_save_utils import root_dir
from addmo.util.experiment_logger import LocalLogger
from addmo.util.experiment_logger import WandbLogger
from addmo.util.experiment_logger import ExperimentLogger
from addmo.s2_data_tuning.config.data_tuning_config import DataTuningFixedConfig
from addmo.s2_data_tuning.data_tuner_fixed import DataTunerByConfig
from addmo.util.load_save import load_data
from addmo.util.plotting_utils import save_pdf
from addmo.util.data_handling import split_target_features
from addmo.s5_insights.model_plots.time_series import plot_timeseries_combined
from addmo.util.load_save import load_config_from_json

def _exe_data_tuning(config, user_input='y'):
    """
    Execute the system_data tuning process in a fixed manner.
    """

    # Configure the logger
    LocalLogger.active = True
    if LocalLogger.active:
        LocalLogger.directory = results_dir_data_tuning(config,user_input)

    WandbLogger.project = "addmo-tests_data_tuning_fixed"
    WandbLogger.active = False
    if WandbLogger.active:
        WandbLogger.directory = results_dir_data_tuning(config,user_input)

    # Initialize logging
    ExperimentLogger.start_experiment(config=config)

    # Create the system_data tuner
    tuner = DataTunerByConfig(config=config)

    # Load the system_data
    xy_raw = load_data(config.abs_path_to_data)
    ExperimentLogger.log({"xy_testraw": xy_raw.iloc[[0, 1, 2, -3, -2, -1]]})

    # Split the system_data
    x, y = split_target_features(config.name_of_target, xy_raw)

    # Tune the system_data
    tuned_x = tuner.tune_fixed(xy_raw)
    ExperimentLogger.log({"x_tuned": tuned_x.iloc[[0, 1, 2, -3, -2, -1]]})

    # Merge target and features
    xy_tuned = tuned_x.join(y)
    ExperimentLogger.log({"xy_tuned": xy_tuned.iloc[[0, 1, 2, -3, -2, -1]]})

    # Drop NaNs
    xy_tuned = xy_tuned.dropna()
    ExperimentLogger.log({"xy_tuned": xy_tuned.iloc[[0, 1, 2, -3, -2, -1]]})

    # Log the tuned system_data
    file_name = 'tuned_xy_fixed'
    ExperimentLogger.log_artifact(xy_tuned, file_name, art_type='system_data')

    # Finish logging
    ExperimentLogger.finish_experiment()

    # Return file paths for plotting data
    saved_data_path = os.path.join(LocalLogger.directory, file_name + '.csv')
    data = pd.read_csv(saved_data_path, delimiter=",", index_col=0, encoding="latin1", header=0)
    config_path = os.path.join(LocalLogger.directory, "config.json")
    with open(config_path, 'r') as f:
        plot_config = json.load(f)

    # Plot tuned data

    figures = plot_timeseries_combined(plot_config,data)
    for fig in figures:
        fig.show()
    os.makedirs(LocalLogger.directory, exist_ok=True)
    for idx, fig in enumerate(figures):
        suffix = "_2weeks" if idx == 1 else ""
        plot_path = os.path.join(LocalLogger.directory, f"{file_name}{suffix}")
        save_pdf(fig, plot_path)

    print("Finished")

def exe_data_tuning_fixed(user_input='y'):
    """Execute the system_data tuning process from a config file."""
    # Path to the config file
    path_to_config = os.path.join(
        root_dir(), 'addmo', 's2_data_tuning', 'config', 'data_tuning_config.json'
    )
    # Create the config object
    config = load_config_from_json(path_to_config, DataTuningFixedConfig)
    # Run data tuning execution
    _exe_data_tuning(config, user_input)


def default_config_exe_data_tuning_fixed(user_input='y'):
    """Execute the system_data tuning process with default config.
    Parameters:
        user_input : str, optional
            If 'y', the contents of the target results directory will be overwritten.
            If 'd', the directory contents will be deleted. Default is 'y'."""
    # Initialize a default config (without loading JSON)
    config = DataTuningFixedConfig()
    # Run data tuning execution
    _exe_data_tuning(config, user_input)

if __name__ == '__main__':
    # Ask the user to overwrite or delete existing results
    user_input = input("To overwrite the existing content type in 'data_tuning_experiment_fixed' results directory <y>, for deleting the current contents type <d>: ")
    # Execute data tuning
    exe_data_tuning_fixed(user_input)
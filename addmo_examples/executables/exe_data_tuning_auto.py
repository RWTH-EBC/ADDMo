import os
import json
import pandas as pd
from addmo.util.plotting import save_pdf
from addmo.util.definitions import results_dir_data_tuning
from addmo.util.load_save_utils import root_dir
from addmo.util.experiment_logger import ExperimentLogger
from addmo.util.experiment_logger import LocalLogger
from addmo.util.experiment_logger import WandbLogger
from addmo.s1_data_tuning_auto.config.data_tuning_auto_config import DataTuningAutoSetup
from addmo.s1_data_tuning_auto.data_tuner_auto import DataTunerAuto
from addmo.s5_insights.model_plots.time_series import plot_timeseries

def exe_data_tuning_auto():
    """
    Execute the system_data tuning process automatically.
    """
    # Path to the config file
    path_to_config = os.path.join(root_dir(), 'addmo', 's1_data_tuning_auto', 'config',
                                'data_tuning_auto_config.json')

    # Create the config object
    # config = load_config_from_json(path_to_config, DataTuningAutoSetup)
    config = DataTuningAutoSetup()

    # Configure the logger
    LocalLogger.directory = results_dir_data_tuning(config)
    LocalLogger.active = True
    WandbLogger.project = "addmo-test_data_auto_tuning"
    WandbLogger.directory = results_dir_data_tuning(config)
    WandbLogger.active = False

    # Initialize logging
    ExperimentLogger.start_experiment(config=config)

    # Create the system_data tuner
    tuner = DataTunerAuto(config=config)

    # Tune the system_data
    tuned_x = tuner.tune_auto()
    y = tuner.y

    tuned_xy = pd.concat([y, tuned_x], axis=1, join="inner").bfill()

    # Log the tuned system_data
    file_name = 'tuned_xy_auto'
    ExperimentLogger.log_artifact(tuned_xy, file_name, art_type='system_data')

    # Return file paths for plotting data
    saved_data_path = os.path.join(LocalLogger.directory, file_name + '.csv')
    config_path = os.path.join(LocalLogger.directory, "config.json")
    with open(config_path, 'r') as f:
        plot_config = json.load(f)

    # Plot tuned data
    plt = plot_timeseries(plot_config, saved_data_path)
    plt.show()
    save_pdf(plt, os.path.join(LocalLogger.directory, file_name ))
    print("Finished")

if __name__ == "__main__":
    exe_data_tuning_auto()
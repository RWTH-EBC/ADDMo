import os

from core.util.definitions import root_dir, results_dir_data_tuning
from core.util.experiment_logger import ExperimentLogger
from core.util.experiment_logger import LocalLogger
from core.util.experiment_logger import WandbLogger
from core.data_tuning_auto.config.data_tuning_auto_config import DataTuningAutoSetup
from core.data_tuning_auto.data_tuner_auto import DataTunerAuto

# Path to the config file
path_to_yaml = os.path.join(root_dir(), 'core', 'data_tuning_auto', 'config',
                            'data_tuning_auto_config.yaml')

# Create the config object
config = DataTuningAutoSetup()

# Load the config from the yaml file
# config.load_yaml_to_class(path_to_yaml)

# Configure the logger
LocalLogger.directory = results_dir_data_tuning(config)
ExperimentLogger.local_logger = LocalLogger
# WandbLogger.project = "todo"
# ExperimentLogger.wandb_logger = WandbLogger

# Initialize logging
ExperimentLogger.start_experiment(config=config)

# Create the data tuner
tuner = DataTunerAuto(config=config)

# Tune the data
tuned_x_data = tuner.tune_auto()

# Drop NaNs
tuned_x_data = tuned_x_data.dropna()

# Log the tuned data
ExperimentLogger.log_artifact(tuned_x_data, name='tuned_data', art_type='data')
#Todo: possibly xy needs to be logged instead of x

print("Finished")

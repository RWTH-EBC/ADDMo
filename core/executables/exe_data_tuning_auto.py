import os

import pandas as pd

from core.util.definitions import root_dir, results_dir_data_tuning
from core.util.experiment_logger import ExperimentLogger
from core.util.experiment_logger import LocalLogger
from core.util.experiment_logger import WandbLogger
from core.s1_data_tuning_auto.config.data_tuning_auto_config import DataTuningAutoSetup
from core.s1_data_tuning_auto.data_tuner_auto import DataTunerAuto

# Path to the config file
path_to_config = os.path.join(root_dir(), 'core', 's1_data_tuning_auto', 'config',
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

# Create the data tuner
tuner = DataTunerAuto(config=config)

# Tune the data
tuned_x = tuner.tune_auto()
y = tuner.y

tuned_xy = pd.concat([y, tuned_x], axis=1, join="inner").bfill()

# Log the tuned data
ExperimentLogger.log_artifact(tuned_xy, name='tuned_xy', art_type='pkl')

print("Finished")

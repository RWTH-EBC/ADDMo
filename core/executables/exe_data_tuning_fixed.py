import os

from core.util.definitions import root_dir, results_dir_data_tuning
from core.util.experiment_logger import LocalLogger
from core.util.experiment_logger import WandbLogger
from core.util.experiment_logger import ExperimentLogger
from core.s2_data_tuning.config.data_tuning_config import DataTuningFixedConfig
from core.s2_data_tuning.data_tuner_fixed import DataTunerByConfig
from core.util.load_save import load_data
from core.util.data_handling import split_target_features


# Path to the config file
path_to_config = os.path.join(root_dir(), 'core', 's2_data_tuning', 'config',
                              'data_tuning_config.json')

# Create the config object
config = DataTuningFixedConfig()
# config = load_config_from_json(path_to_config, DataTuningFixedConfig)

# Configure the logger
LocalLogger.directory = results_dir_data_tuning(config)
LocalLogger.active = False
WandbLogger.project = "addmo-tests_data_tuning_fixed"
WandbLogger.directory = results_dir_data_tuning(config)
WandbLogger.active = True

# Initialize logging
ExperimentLogger.start_experiment(config=config)

# Create the data tuner
tuner = DataTunerByConfig(config=config)

# Load the data
xy_raw = load_data(config.path_to_raw_data)
ExperimentLogger.log({"xy_raw": xy_raw.iloc[[0, 1, 2, -3, -2, -1]]})

# Split the data
x, y = split_target_features(config.target, xy_raw)

# Tune the data
tuned_x = tuner.tune_fixed(xy_raw)
ExperimentLogger.log({"x_tuned": tuned_x.iloc[[0, 1, 2, -3, -2, -1]]})

# Merge target and features
xy_tuned = tuned_x.join(y)
ExperimentLogger.log({"xy_tuned": xy_tuned.iloc[[0, 1, 2, -3, -2, -1]]})

# Drop NaNs
xy_tuned = xy_tuned.dropna()
ExperimentLogger.log({"xy_tuned": xy_tuned.iloc[[0, 1, 2, -3, -2, -1]]})

# Log the tuned data
ExperimentLogger.log_artifact(xy_tuned, name='xy_tuned', art_type='pkl')

# Finish logging
ExperimentLogger.finish_experiment()

print("Finished")
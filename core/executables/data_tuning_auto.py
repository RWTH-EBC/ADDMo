

# load config for automatic data tuning aka GUI config
path_to_config = r"D:\04_GitRepos\addmo-extra\core\config_files\1_datatuning_GUI_config.yaml"
with open(path_to_config, "r") as f:
    config = yaml.safe_load(f)

# create data tuning setup
dt_config = DataTuningSetup()

# load data
df = load_raw_data(dt_config.abs_path_to_data)

# feature construction
x_created_1 = feature_construction.manual_feature_lags(dt_config, df)
x_created_2 = feature_construction.manual_target_lags(dt_config, df)
# x_created_3 = feature_construction.automatic_timeseries_target_lag_constructor(
#     dt_config, df
# # )
# x_created_4 = feature_construction.automatic_feature_lag_constructor(dt_config, df)


import os

from core.util.definitions import root_dir, results_dir_data_tuning_local
from core.util.experiment_logger import LocalLogger
from core.data_tuning_auto.config.data_tuning_auto_config import DataTuningAutoSetup
from core.data_tuning_auto.data_tuner_auto import DataTunerAuto

# Path to the config file
path_to_yaml = os.path.join(root_dir(), 'core', 'data_tuning_auto', 'config',
                            'data_tuning_auto_config.yaml')

# Create the config object
config = DataTuningAutoSetup()

# Load the config from the yaml file
config.load_yaml_to_class(path_to_yaml)

# Create the experiment logger
logger = LocalLogger(directory=results_dir_data_tuning_local(config))

logger.start_experiment(config=config) # actually not necessary for local logger

# Create the data tuner
tuner = DataTunerByConfig(config=config, logger=logger)

# Tune the data
tuned_x_data = tuner.tune_fixed()

# Drop NaNs
tuned_x_data = tuned_x_data.dropna()

# Log the tuned data
logger.log_artifact(tuned_x_data, name='tuned_data', art_type='data')
#Todo: possibly xy needs to be logged instead of x

print("Finished")

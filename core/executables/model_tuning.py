import os

from core.util.definitions import root_dir, results_dir_model_tuning_local
from core.util.experiment_logger import LocalLogger

from core.model_tuning.config.model_tuning_config import ModelTuningSetup
from core.model_tuning.model_tuner import ModelTuner

# Path to the config file
path_to_yaml = os.path.join(root_dir(), 'core', 'model_tuning', 'config',
                            'model_tuning_config.yaml')

# Create the config object
config = ModelTuningSetup()

# Load the config from the yaml file
config.load_yaml_to_class(path_to_yaml)

# Create the experiment logger
logger = LocalLogger(directory=results_dir_model_tuning_local(config))

logger.start_experiment(config=config) # actually not necessary for local logger

# Create the model tuner
model_tuner = ModelTuner(config=config, logger=logger)

# Tune the model
best_model = model_tuner.tune_model()

# Log the tuned model
logger.log_artifact(best_model, name='tuned_model', art_type='model')


print("Finished")

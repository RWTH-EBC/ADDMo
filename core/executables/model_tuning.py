import os

from core.util.definitions import root_dir, results_dir_model_tuning_local
from core.util.experiment_logger import LocalLogger
from core.util.experiment_logger import WandbLogger
from core.util.experiment_logger import ExperimentLogger

from core.model_tuning.config.model_tuning_config import ModelTuningSetup
from core.model_tuning.model_tuner import ModelTuner

# Path to the config file
path_to_yaml = os.path.join(root_dir(), 'core', 'model_tuning', 'config',
                            'model_tuning_config.yaml')

# Create the config object
config = ModelTuningSetup()

# Load the config from the yaml file
config.load_yaml_to_class(path_to_yaml)

# Configure the logger
LocalLogger.directory = results_dir_model_tuning_local(config)
ExperimentLogger.local_logger = LocalLogger
# WandbLogger.project = "todo"
# ExperimentLogger.wandb_logger = WandbLogger

# Initialize logging
ExperimentLogger.start_experiment(config=config)

# Create the model tuner
model_tuner = ModelTuner(config=config)

# Tune the models
model_dict = model_tuner.tune_all_models()

# Get the best model
best_model = model_tuner.get_best_model(model_dict)

# Log the tuned model
ExperimentLogger.log_artifact(best_model, name='tuned_model', art_type='model')


print("Finished")

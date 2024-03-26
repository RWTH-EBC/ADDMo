import os
import pandas as pd

from core.util.definitions import root_dir, results_dir_model_tuning
from core.util.experiment_logger import LocalLogger
from core.util.experiment_logger import WandbLogger
from core.util.experiment_logger import ExperimentLogger

from core.s3_model_tuning.config.model_tuning_config import ModelTuningExperimentConfig
from core.s3_model_tuning.model_tuner import ModelTuner
from core.util.load_save import load_data
from core.util.data_handling import split_target_features
def exe_model_tuning(config=None):
    # Configure the logger
    LocalLogger.directory = results_dir_model_tuning(config)
    LocalLogger.active = False
    WandbLogger.project = "addmo-test_model_tuning"
    WandbLogger.directory = results_dir_model_tuning(config)
    WandbLogger.active = True

    # Initialize logging
    ExperimentLogger.start_experiment(config=config)

    # Create the model tuner
    model_tuner = ModelTuner(config=config)

    # Load the data
    xy_tuned = load_data(config.abs_path_to_data)

    # Select training and validation period
    xy_tuned_train_val = xy_tuned.loc[config.start_train_val:config.stop_train_val]
    x_train_val, y_train_val = split_target_features(config.name_of_target, xy_tuned_train_val)

    # log start and end of the data
    ExperimentLogger.log({"xy_tuned_train_val": pd.concat([xy_tuned_train_val.head(5), xy_tuned_train_val.tail(5)])})

    # Tune the models
    model_dict = model_tuner.tune_all_models(x_train_val, y_train_val)

    # Get the best model
    best_model = model_tuner.get_best_model(model_dict)

    # Log the best model
    ExperimentLogger.log_artifact(best_model, name='best_model', art_type='onnx')


    print("Finished")

if __name__ == "__main__":
    # Path to the config file
    path_to_yaml = os.path.join(root_dir(), 'core', 's3_model_tuning', 'config',
                                'model_tuning_config.yaml')

    # Create the config object
    config = ModelTuningExperimentConfig()

    # Load the config from the yaml file
    # config.load_yaml_to_class(path_to_yaml)
    exe_model_tuning(config)
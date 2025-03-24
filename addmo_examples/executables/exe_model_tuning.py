import os
import pandas as pd

from addmo.util.definitions import results_dir_model_tuning, results_dir_data_tuning_auto
from addmo.util.load_save_utils import root_dir
from addmo.util.experiment_logger import LocalLogger
from addmo.util.experiment_logger import WandbLogger
from addmo.util.experiment_logger import ExperimentLogger

from addmo.s3_model_tuning.config.model_tuning_config import ModelTuningExperimentConfig
from addmo.s3_model_tuning.config.model_tuning_config import ModelTunerConfig
from addmo.s3_model_tuning.model_tuner import ModelTuner
from addmo.util.load_save import load_data
from addmo.util.load_save import load_config_from_json
from addmo.util.data_handling import split_target_features

def exe_model_tuning(config=None):
    """
    Executes model tuning process and returns the best model.
    """

    # Configure the logger
    LocalLogger.directory = results_dir_model_tuning(config)
    LocalLogger.active = True
    WandbLogger.project = "addmo-test_model_tuning"
    WandbLogger.directory = results_dir_model_tuning(config)
    WandbLogger.active = False

    # Initialize logging
    ExperimentLogger.start_experiment(config=config)

    # Create the model tuner
    model_tuner = ModelTuner(config=config.config_model_tuner)

    # Load the system_data
    tuned_data_path= results_dir_data_tuning_auto(config=config)
    xy_tuned = pd.read_csv(tuned_data_path, delimiter=",", index_col=[0], encoding="latin1", header=[0] )

    # Select training and validation period
    xy_tuned_train_val = xy_tuned.loc[config.start_train_val:config.stop_train_val]
    x_train_val, y_train_val = split_target_features(config.name_of_target, xy_tuned_train_val)

    # log start and end of the system_data
    ExperimentLogger.log({"xy_tuned_train_val": pd.concat([xy_tuned_train_val.head(5), xy_tuned_train_val.tail(5)])})

    # Tune the models
    model_dict = model_tuner.tune_all_models(x_train_val, y_train_val)

    # Get the best model
    best_model_name = model_tuner.get_best_model_name(model_dict)
    best_model = model_tuner.get_model(model_dict, best_model_name)

    # Log the best model
    ExperimentLogger.log_artifact(best_model, name='best_model', art_type='keras')


    print("Finished")

if __name__ == "__main__":
    # Create the config object
    config = ModelTuningExperimentConfig()

    config.config_model_tuner.validation_score_splitting = 'UnivariateSplitter'
    config.config_model_tuner.validation_score_splitting_kwargs = None

    # run
    exe_model_tuning(config)
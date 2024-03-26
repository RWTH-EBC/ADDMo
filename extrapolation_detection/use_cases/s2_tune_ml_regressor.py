import os

import pandas as pd

import extrapolation_detection.util.loading_saving
from core.s3_model_tuning.models.abstract_model import AbstractMLModel
from core.s3_model_tuning.model_tuner import ModelTuner
from core.util.data_handling import split_target_features

from extrapolation_detection.util import data_handling
from extrapolation_detection.use_cases.config.ed_experiment_config import (
    ExtrapolationExperimentConfig,
)


def exe_tune_regressor(config: ExtrapolationExperimentConfig):
    xy_training = extrapolation_detection.util.loading_saving.read_csv("xy_train", directory=config.experiment_folder)
    xy_validation = extrapolation_detection.util.loading_saving.read_csv("xy_val", directory=config.experiment_folder)

    # Create the config object
    config_MT = config.config_model_tuning

    # training data of extrapolation experiment is used for model tuning
    xy_train_val = pd.concat([xy_training, xy_validation])
    x_train_val, y_train_val = split_target_features(
        config.name_of_target, xy_train_val
    )

    model_tuner = ModelTuner(config=config_MT)
    model_dict = model_tuner.tune_all_models(x_train_val, y_train_val)
    best_model = model_tuner.get_best_model(model_dict)
    regressor: AbstractMLModel = best_model

    # safe regressor to pickle #Todo: evtl. via onnx?
    regressor_directory = os.path.join(config.experiment_folder, "regressors")
    extrapolation_detection.util.loading_saving.write_pkl(regressor, "regressor", directory=regressor_directory)

    extrapolation_detection.util.loading_saving.write_csv(
        xy_train_val, "xy_regressor_fit", directory=regressor_directory
    )
    extrapolation_detection.util.loading_saving.write_csv(
        x_train_val, "x_regressor_fit", directory=regressor_directory
    )
    extrapolation_detection.util.loading_saving.write_csv(
        y_train_val, "y_regressor_fit", directory=regressor_directory
    )


if __name__ == "__main__":
    config = ExtrapolationExperimentConfig()
    exe_tune_regressor(config)

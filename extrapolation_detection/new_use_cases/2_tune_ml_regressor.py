import os

import pandas as pd

from core.util.definitions import root_dir
from core.s3_model_tuning.models.abstract_model import AbstractMLModel
from core.s3_model_tuning.model_tuner import ModelTuner
from core.s3_model_tuning.config.model_tuning_config import ModelTuningSetup
from core.util.data_handling import split_target_features

from extrapolation_detection.machine_learning_util import data_handling
from extrapolation_detection.new_use_cases.ed_experiment_config import (
    ExtrapolationExperimentConfig,
)


def exe_tune_regressor(config: ExtrapolationExperimentConfig):
    xy_training = data_handling.read_csv(
        "xy_train", directory=config.experiment_name
    )
    xy_validation = data_handling.read_csv(
        "xy_val", directory=config.experiment_name
    )

    # Create the config object
    config_MT = config.model_tuning_config

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
    data_handling.write_pkl(
        regressor,
        "regressor",
        directory=os.path.join(config.experiment_name, "regressors"),
        override=True,
    )


if __name__ == "__main__":
    config = ExtrapolationExperimentConfig()
    exe_tune_regressor(config)

import os

import pandas as pd

from extrapolation_detection.util import loading_saving
from core.s3_model_tuning.models.abstract_model import AbstractMLModel
from core.s3_model_tuning.model_tuner import ModelTuner
from core.util.data_handling import split_target_features

from extrapolation_detection.util import data_handling
from extrapolation_detection.use_cases.config.ed_experiment_config import (
    ExtrapolationExperimentConfig,
)


def exe(config: ExtrapolationExperimentConfig):
    xy_training = loading_saving.read_csv(
        "xy_train", directory=config.experiment_folder
    )
    xy_validation = loading_saving.read_csv(
        "xy_val", directory=config.experiment_folder
    )

    # training data of extrapolation experiment is used for model tuning
    xy_train_val = pd.concat([xy_training, xy_validation])
    x_train_val, y_train_val = split_target_features(
        config.name_of_target, xy_train_val
    )

    model_tuner = ModelTuner(config=config.config_model_tuning)
    model_dict = model_tuner.tune_all_models(x_train_val, y_train_val)
    best_model_name = model_tuner.get_best_model_name(model_dict)
    best_model = model_tuner.get_model(model_dict, best_model_name)
    best_model_val_score = model_tuner.get_model_validation_score(
        model_dict, best_model_name
    )
    best_model_params = best_model.get_params()
    regressor: AbstractMLModel = best_model

    # generate prediction for fit period
    y_pred = pd.Series(regressor.predict(x_train_val), index=x_train_val.index)
    y_pred.name = config.name_of_target + "_pred"

    # safe regressor to pickle #Todo: evtl. via onnx?
    regressor_directory = os.path.join(config.experiment_folder, "regressors")
    loading_saving.write_pkl(regressor, "regressor", directory=regressor_directory)

    loading_saving.write_csv(xy_train_val, "xy_regressor_fit", directory=regressor_directory)
    loading_saving.write_csv(x_train_val, "x_regressor_fit", directory=regressor_directory)
    loading_saving.write_csv(y_train_val, "y_regressor_fit", directory=regressor_directory)
    loading_saving.write_csv(y_pred, "pred_regressor_fit", directory=regressor_directory)

    # log model infos
    model_infos = pd.DataFrame([best_model_params])
    model_infos.loc[0, "best_model_name"] = best_model_name
    model_infos.loc[0, "best_model_val_score"] = best_model_val_score
    loading_saving.write_csv(model_infos, "model_infos", directory=regressor_directory)

    print(f"{__name__} executed")


if __name__ == "__main__":
    config = ExtrapolationExperimentConfig()
    exe(config)

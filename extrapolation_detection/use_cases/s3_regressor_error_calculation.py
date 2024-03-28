import os

import pandas as pd

from extrapolation_detection.util import loading_saving
from core.s3_model_tuning.models.abstract_model import AbstractMLModel
from core.util.data_handling import split_target_features

from extrapolation_detection.util import data_handling
from extrapolation_detection.n_D_extrapolation.score_regressor_per_data_point import (
    score_per_sample,
)
from extrapolation_detection.use_cases.config.ed_experiment_config import (
    ExtrapolationExperimentConfig,
)


def exe(config: ExtrapolationExperimentConfig):
    # load model
    regressor: AbstractMLModel = loading_saving.read_pkl(
        "regressor", directory=os.path.join(config.experiment_folder, "regressors")
    )

    xy_training = loading_saving.read_csv(
        "xy_train", directory=config.experiment_folder
    )
    xy_validation = loading_saving.read_csv(
        "xy_val", directory=config.experiment_folder
    )
    xy_test = loading_saving.read_csv("xy_test", directory=config.experiment_folder)
    xy_remaining = loading_saving.read_csv(
        "xy_remaining", directory=config.experiment_folder
    )
    xy_grid = loading_saving.read_csv("xy_grid", directory=config.experiment_folder)

    x_train, y_train = split_target_features(config.name_of_target, xy_training)
    errors_train = score_per_sample(
        regressor, x_train, y_train, metric=config.true_outlier_threshold_error_metric
    )
    loading_saving.write_csv(
        errors_train, "errors_train", directory=config.experiment_folder
    )

    x_val, y_val = split_target_features(config.name_of_target, xy_validation)
    errors_val = score_per_sample(
        regressor, x_val, y_val, metric=config.true_outlier_threshold_error_metric
    )
    loading_saving.write_csv(
        errors_val, "errors_val", directory=config.experiment_folder
    )

    x_test, y_test = split_target_features(config.name_of_target, xy_test)
    errors_test = score_per_sample(
        regressor, x_test, y_test, metric=config.true_outlier_threshold_error_metric
    )

    loading_saving.write_csv(
        errors_test, "errors_test", directory=config.experiment_folder
    )

    x_remaining, y_remaining = split_target_features(
        config.name_of_target, xy_remaining
    )
    errors_remaining = score_per_sample(
        regressor,
        x_remaining,
        y_remaining,
        metric=config.true_outlier_threshold_error_metric,
    )
    loading_saving.write_csv(
        errors_remaining, "errors_remaining", directory=config.experiment_folder
    )

    x_grid, y_grid = split_target_features(config.name_of_target, xy_grid)
    errors_grid = score_per_sample(
        regressor, x_grid, y_grid, metric=config.true_outlier_threshold_error_metric
    )
    loading_saving.write_csv(
        errors_grid, "errors_grid", directory=config.experiment_folder
    )

    # calc mean errors and save them in one csv with index indicating the period
    mean_errors = pd.DataFrame(
        {
            "mean_errors_train": errors_train["error"].mean(),
            "mean_errors_val": errors_val["error"].mean(),
            "mean_errors_test": errors_test["error"].mean(),
            "mean_errors_remaining": errors_remaining["error"].mean(),
            "mean_errors_grid": errors_grid["error"].mean(),
        },
        index=[config.true_outlier_threshold_error_metric],
    )

    loading_saving.write_csv(mean_errors, "mean_errors", directory=config.experiment_folder)





    print(f"{__name__} executed")


if __name__ == "__main__":
    config = ExtrapolationExperimentConfig()
    exe(config)

import os

from core.s3_model_tuning.models.abstract_model import AbstractMLModel
from core.util.data_handling import split_target_features

from extrapolation_detection.util import data_handling
from extrapolation_detection.n_D_extrapolation.score_regressor_per_data_point import (
    score_per_sample,
)
from extrapolation_detection.use_cases.config.ed_experiment_config import (
    ExtrapolationExperimentConfig,
)


def exe_regressor_error_calculation(config: ExtrapolationExperimentConfig):
    # load model
    regressor: AbstractMLModel = data_handling.read_pkl(
        "regressor", directory=os.path.join(config.experiment_folder, "regressors")
    )

    xy_training = data_handling.read_csv("xy_train", directory=config.experiment_folder)
    xy_validation = data_handling.read_csv("xy_val", directory=config.experiment_folder)
    xy_test = data_handling.read_csv("xy_test", directory=config.experiment_folder)
    xy_remaining = data_handling.read_csv(
        "xy_remaining", directory=config.experiment_folder
    )
    xy_grid = data_handling.read_csv("xy_grid", directory=config.experiment_folder)

    x_train, y_train = split_target_features(config.name_of_target, xy_training)
    errors_train = score_per_sample(regressor, x_train, y_train, metric="mae")
    data_handling.write_csv(
        errors_train, "errors_train", directory=config.experiment_folder
    )

    x_val, y_val = split_target_features(config.name_of_target, xy_validation)
    errors_val = score_per_sample(regressor, x_val, y_val, metric="mae")
    data_handling.write_csv(
        errors_val, "errors_val", directory=config.experiment_folder
    )

    x_test, y_test = split_target_features(config.name_of_target, xy_test)
    errors_test = score_per_sample(regressor, x_test, y_test, metric="mae")
    data_handling.write_csv(
        errors_test, "errors_test", directory=config.experiment_folder
    )

    x_remaining, y_remaining = split_target_features(
        config.name_of_target, xy_remaining
    )
    errors_remaining = score_per_sample(
        regressor, x_remaining, y_remaining, metric="mae"
    )
    data_handling.write_csv(
        errors_remaining, "errors_remaining", directory=config.experiment_folder
    )

    x_grid, y_grid = split_target_features(config.name_of_target, xy_grid)
    errors_grid = score_per_sample(regressor, x_grid, y_grid, metric="mae")
    data_handling.write_csv(
        errors_grid, "errors_grid", directory=config.experiment_folder
    )


if __name__ == "__main__":
    config = ExtrapolationExperimentConfig()
    exe_regressor_error_calculation(config)

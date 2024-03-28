import os

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


def exe_regressor_error_calculation(config: ExtrapolationExperimentConfig):
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


    print(f"{__name__} executed")


if __name__ == "__main__":
    config = ExtrapolationExperimentConfig()
    exe_regressor_error_calculation(config)

import os

import pandas as pd

from core.util.data_handling import split_target_features

from extrapolation_detection.machine_learning_util import data_handling
from extrapolation_detection.detector import scoring
from extrapolation_detection.new_use_cases.ed_experiment_config import (
    ExtrapolationExperimentConfig,
)
from extrapolation_detection.detector.detectors import AbstractDetector


def exe_detector_score_calculation(config: ExtrapolationExperimentConfig):
    # load data
    xy_training = data_handling.read_csv("xy_train", directory=config.experiment_name)
    xy_validation = data_handling.read_csv("xy_val", directory=config.experiment_name)
    xy_test = data_handling.read_csv("xy_test", directory=config.experiment_name)
    xy_remaining = data_handling.read_csv(
        "xy_remaining", directory=config.experiment_name
    )
    xy_grid = data_handling.read_csv("xy_grid", directory=config.experiment_name)

    true_validity_remaining = data_handling.read_csv(
        "true_validity_remaining", directory=config.experiment_name
    ).squeeze()
    true_valitidy_grid = data_handling.read_csv(
        "true_validity_grid", directory=config.experiment_name
    ).squeeze()

    # score all detectors that are saved in the directory
    directory = os.path.join(config.experiment_name, "detectors")
    for detector_file in os.listdir(directory):
        if detector_file.endswith(".pkl"):
            # load detector
            detector: AbstractDetector = data_handling.read_pkl(
                detector_file, directory=directory
            )
            # get name without file ending
            detector_name = detector_file.split(".")[0]

            # score the detector
            x_train, y_train = split_target_features(config.name_of_target, xy_training)
            n_score_train = detector.score(x_train.values)
            n_score_train_df = pd.DataFrame(n_score_train, index=x_train.index)
            data_handling.write_csv(
                n_score_train_df,
                f"n_score_train_{detector_name}",
                directory=config.experiment_name,
            )

            x_val, y_val = split_target_features(config.name_of_target, xy_validation)
            n_score_val = detector.score(x_val.values)
            n_score_val_df = pd.DataFrame(n_score_val, index=x_val.index)
            data_handling.write_csv(
                n_score_val_df,
                f"n_score_val_{detector_name}",
                directory=config.experiment_name,
            )

            x_test, y_test = split_target_features(config.name_of_target, xy_test)
            n_score_test = detector.score(x_test.values)
            n_score_test_df = pd.DataFrame(n_score_test, index=x_test.index)
            data_handling.write_csv(
                n_score_test_df,
                f"n_score_test_{detector_name}",
                directory=config.experiment_name,
            )

            x_remaining, y_remaining = split_target_features(
                config.name_of_target, xy_remaining
            )
            n_score_remaining = detector.score(x_remaining.values)
            n_score_remaining_df = pd.DataFrame(
                n_score_remaining, index=x_remaining.index
            )
            data_handling.write_csv(
                n_score_remaining_df,
                f"n_score_remaining_{detector_name}",
                directory=config.experiment_name,
            )

            x_grid, y_grid = split_target_features(config.name_of_target, xy_grid)
            n_score_grid = detector.score(x_grid.values)
            n_score_grid_df = pd.DataFrame(n_score_grid, index=x_grid.index)
            data_handling.write_csv(
                n_score_grid_df,
                f"n_score_grid_{detector_name}",
                directory=config.experiment_name,
            )

            # calculate the evaluation of the detector
            novelty_threshold = data_handling.read_csv(
                f"{detector_name}_threshold", directory=directory
            ).iloc[0, 0]
            detector_evaluation_remaining = scoring.score_samples(
                true_validity_remaining.values.reshape(-1, 1),
                n_score_remaining_df.values.reshape(-1, 1),
                novelty_threshold,
                beta=config.detector_config.beta_f_score,
            )
            detector_evaluation_remaining_df = pd.DataFrame.from_dict(detector_evaluation_remaining, orient="index")
            data_handling.write_csv(
                detector_evaluation_remaining_df,
                f"detector_evaluation_remaining_{detector_name}",
                directory=config.experiment_name,
            )

            detector_evaluation_grid = scoring.score_samples(
                true_valitidy_grid.values.reshape(-1, 1),
                n_score_grid_df.values.reshape(-1, 1),
                novelty_threshold,
                beta=config.detector_config.beta_f_score,
            )
            detector_evaluation_grid_df = pd.DataFrame.from_dict(detector_evaluation_grid, orient="index")
            data_handling.write_csv(
                detector_evaluation_grid_df,
                f"detector_evaluation_grid_{detector_name}",
                directory=config.experiment_name,
            )


if __name__ == "__main__":
    config = ExtrapolationExperimentConfig()
    exe_detector_score_calculation(config)

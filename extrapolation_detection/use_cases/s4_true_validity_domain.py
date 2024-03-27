import pandas as pd

import extrapolation_detection.util.loading_saving
from extrapolation_detection.util import data_handling
from extrapolation_detection.use_cases.config.ed_experiment_config import (
    ExtrapolationExperimentConfig,
)
from extrapolation_detection.n_D_extrapolation import true_validity_domain


def exe_true_validity_domain(config: ExtrapolationExperimentConfig):
    """Classifies the data points to be in the true validity domain or not based on the
    regressor-error."""

    errors_train = extrapolation_detection.util.loading_saving.read_csv(
        "errors_train", directory=config.experiment_folder
    )
    errors_val = extrapolation_detection.util.loading_saving.read_csv("errors_val", directory=config.experiment_folder)
    errors_test = extrapolation_detection.util.loading_saving.read_csv(
        "errors_test", directory=config.experiment_folder
    )
    errors_remaining = extrapolation_detection.util.loading_saving.read_csv(
        "errors_remaining", directory=config.experiment_folder
    )
    errors_grid = extrapolation_detection.util.loading_saving.read_csv(
        "errors_grid", directory=config.experiment_folder
    )

    # infer threshold
    errors_train_val_test = pd.concat([errors_train, errors_val, errors_test])
    true_validity_threshold = true_validity_domain.infer_threshold(
        config.true_outlier_fraction, errors_train_val_test["error"]
    )


    # classify errors to validity boolean
    true_validity_train = true_validity_domain.classify_errors_2_true_validity(
        errors_train["error"], true_validity_threshold
    )
    true_validity_val = true_validity_domain.classify_errors_2_true_validity(
        errors_val["error"], true_validity_threshold
    )
    true_validity_test = true_validity_domain.classify_errors_2_true_validity(
        errors_test["error"], true_validity_threshold
    )
    true_validity_remaining = true_validity_domain.classify_errors_2_true_validity(
        errors_remaining["error"], true_validity_threshold
    )
    true_validity_grid = true_validity_domain.classify_errors_2_true_validity(
        errors_grid["error"], true_validity_threshold
    )

    # Save to csv
    # to dataframe for saving in human readable csv format
    true_validity_threshold = pd.DataFrame([true_validity_threshold])
    extrapolation_detection.util.loading_saving.write_csv(
        true_validity_threshold, "true_validity_threshold", directory=config.experiment_folder
    )

    extrapolation_detection.util.loading_saving.write_csv(
        true_validity_train, "true_validity_train", directory=config.experiment_folder
    )
    extrapolation_detection.util.loading_saving.write_csv(
        true_validity_val, "true_validity_val", directory=config.experiment_folder
    )
    extrapolation_detection.util.loading_saving.write_csv(
        true_validity_test, "true_validity_test", directory=config.experiment_folder
    )
    extrapolation_detection.util.loading_saving.write_csv(
        true_validity_remaining,
        "true_validity_remaining",
        directory=config.experiment_folder,
    )
    extrapolation_detection.util.loading_saving.write_csv(
        true_validity_grid, "true_validity_grid", directory=config.experiment_folder
    )

    print(f"{__name__} executed")


if __name__ == "__main__":
    config = ExtrapolationExperimentConfig()
    exe_true_validity_domain(config)

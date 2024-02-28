import os

import pandas as pd

from extrapolation_detection.machine_learning_util import data_handling
from extrapolation_detection.new_use_cases.ed_experiment_config import (
    ExtrapolationExperimentConfig,
)
from extrapolation_detection.n_D_extrapolation import true_validity_domain


def exe_true_validity_domain(config: ExtrapolationExperimentConfig):
    """Classifies the data points to be in the true validity domain or not based on the
    regressor-error."""

    errors_train = data_handling.read_csv(
        "errors_train", directory=config.experiment_name
    )
    errors_val = data_handling.read_csv("errors_val", directory=config.experiment_name)
    errors_test = data_handling.read_csv(
        "errors_test", directory=config.experiment_name
    )
    errors_remaining = data_handling.read_csv(
        "errors_remaining", directory=config.experiment_name
    )

    # infer threshold
    errors_train_val_test = pd.concat([errors_train, errors_val, errors_test])
    absolute_threshold = true_validity_domain.infer_threshold(
        config.true_outlier_fraction, errors_train_val_test["error"]
    )

    # classify errors to validity boolean
    true_validity_train = true_validity_domain.classify_errors_2_true_validity(
        errors_train["error"], absolute_threshold
    )
    true_validity_val = true_validity_domain.classify_errors_2_true_validity(
        errors_val["error"], absolute_threshold
    )
    true_validity_test = true_validity_domain.classify_errors_2_true_validity(
        errors_test["error"], absolute_threshold
    )
    true_validity_remaining = true_validity_domain.classify_errors_2_true_validity(
        errors_remaining["error"], absolute_threshold
    )

    # Save to csv
    data_handling.write_csv(
        true_validity_train, "true_validity_train", directory=config.experiment_name
    )
    data_handling.write_csv(
        true_validity_val, "true_validity_val", directory=config.experiment_name
    )
    data_handling.write_csv(
        true_validity_test, "true_validity_test", directory=config.experiment_name
    )
    data_handling.write_csv(
        true_validity_remaining,
        "true_validity_remaining",
        directory=config.experiment_name,
    )


if __name__ == "__main__":
    config = ExtrapolationExperimentConfig()
    exe_true_validity_domain(config)

import pandas as pd

from addmo.util.experiment_logger import ExperimentLogger
from aixtra.util import loading_saving_aixtra
from aixtra_use_case.config.ed_experiment_config import (
    ExtrapolationExperimentConfig,
)
from aixtra.extrapolation_detection.n_D_extrapolation import true_validity_domain


def exe(config: ExtrapolationExperimentConfig):
    """Classifies the system_data points to be in the true validity domain or not based on the
    regressor-error."""

    errors_train = loading_saving_aixtra.read_csv(
        "errors_train", directory=config.experiment_folder
    )
    errors_val = loading_saving_aixtra.read_csv("errors_val", directory=config.experiment_folder)
    errors_test = loading_saving_aixtra.read_csv(
        "errors_test", directory=config.experiment_folder
    )
    errors_remaining = loading_saving_aixtra.read_csv(
        "errors_remaining", directory=config.experiment_folder
    )
    errors_grid = loading_saving_aixtra.read_csv(
        "errors_grid", directory=config.experiment_folder
    )

    # if needed, infer threshold
    errors_train_val_test = pd.concat([errors_train, errors_val, errors_test])
    if config.true_outlier_threshold is None:
        true_validity_threshold = true_validity_domain.infer_threshold(
            config.true_outlier_fraction, errors_train_val_test["error"]
        )
    else:
        true_validity_threshold = config.true_outlier_threshold


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

    # warn about to few true valid system_data points
    true_valid_points = true_validity_train.sum() + true_validity_val.sum()
    if true_valid_points < 20:
        print(f"Warning: Only {true_valid_points} true valid system_data points in training and validation set")


    # Save to csv
    # to dataframe for saving in human readable csv format
    true_validity_threshold = pd.DataFrame([true_validity_threshold])
    loading_saving_aixtra.write_csv(
        true_validity_threshold, "true_validity_threshold", directory=config.experiment_folder
    )

    loading_saving_aixtra.write_csv(
        true_validity_train, "true_validity_train", directory=config.experiment_folder
    )
    loading_saving_aixtra.write_csv(
        true_validity_val, "true_validity_val", directory=config.experiment_folder
    )
    loading_saving_aixtra.write_csv(
        true_validity_test, "true_validity_test", directory=config.experiment_folder
    )
    loading_saving_aixtra.write_csv(
        true_validity_remaining,
        "true_validity_remaining",
        directory=config.experiment_folder,
    )
    loading_saving_aixtra.write_csv(
        true_validity_grid, "true_validity_grid", directory=config.experiment_folder
    )

    # calc true validity fraction (mean of true validity) and save them in one csv with index indicating the period
    true_valid_fraction_dict = {
        "true_valid_fraction_train": true_validity_train.mean(),
        "true_valid_fraction_val": true_validity_val.mean(),
        "true_valid_fraction_test": true_validity_test.mean(),
        "true_valid_fraction_remaining": true_validity_remaining.mean(),
        "true_valid_fraction_grid": true_validity_grid.mean(),
    }
    true_valid_fraction = pd.DataFrame(
        true_valid_fraction_dict,
        index=[config.true_outlier_threshold_error_metric],
    )

    loading_saving_aixtra.write_csv(
        true_valid_fraction, "true_valid_fraction", directory=config.experiment_folder
    )

    ExperimentLogger.log(true_valid_fraction_dict)

    # log true_validity_threshold
    ExperimentLogger.log({"true_validity_threshold": true_validity_threshold.iloc[0,0]})

    print(f"{__name__} executed")


if __name__ == "__main__":
    config = ExtrapolationExperimentConfig()
    exe(config)

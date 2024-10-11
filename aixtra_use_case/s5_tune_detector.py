import os

import pandas as pd

from aixtra.util import loading_saving_aixtra, data_handling
from aixtra_use_case.config.ed_experiment_config import (
    ExtrapolationExperimentConfig,
)
from aixtra.extrapolation_detection.detector.tuning_mechanisms import (
    tune_detector,
    untuned_detector,
)
from addmo.util.data_handling import split_target_features


def exe(config: ExtrapolationExperimentConfig):
    xy_training = loading_saving_aixtra.read_csv("xy_train", directory=config.experiment_folder)
    xy_validation = loading_saving_aixtra.read_csv("xy_val", directory=config.experiment_folder)
    xy_test = loading_saving_aixtra.read_csv("xy_test", directory=config.experiment_folder)
    xy_remaining = loading_saving_aixtra.read_csv(
        "xy_remaining", directory=config.experiment_folder
    )

    true_validity_train = loading_saving_aixtra.read_csv(
        "true_validity_train", directory=config.experiment_folder
    ).squeeze()
    true_validity_val = loading_saving_aixtra.read_csv(
        "true_validity_val", directory=config.experiment_folder
    ).squeeze()
    true_validity_test = loading_saving_aixtra.read_csv(
        "true_validity_test", directory=config.experiment_folder
    ).squeeze()
    true_validity_remaining = loading_saving_aixtra.read_csv(
        "true_validity_remaining", directory=config.experiment_folder
    ).squeeze()

    # regressor uses same system_data for fit as detector
    xy_detector_fit = pd.concat([xy_training, xy_validation])
    true_validity_detector_fit = pd.concat([true_validity_train, true_validity_val])

    # move true invalid system_data from training to validation (methodology)
    (
        xy_detector_fit,
        xy_detector_val,
        true_validity_detector_fit,
        true_validity_detector_val,
    ) = data_handling.move_true_invalid_from_training_2_validation(
        xy_detector_fit, xy_test, true_validity_detector_fit, true_validity_test
    )

    # handle different option for the validation set
    tag = "test"
    if config.config_detector.use_train_for_validation:
        xy_detector_val = pd.concat([xy_detector_val, xy_detector_fit])
        true_validity_detector_val = pd.concat([true_validity_detector_val, true_validity_detector_fit])
        tag = tag + "+fit"
    if config.config_detector.use_remaining_for_validation:
        xy_detector_val = pd.concat([xy_detector_val, xy_remaining])
        true_validity_detector_val = pd.concat(
            [true_validity_detector_val, true_validity_remaining]
        )
        tag = tag + "+remaining"

    if tag == "test+fit+remaining":
        tag = "ideal"

    # delete target, as it is not needed for the detector
    x_detector_fit, _ = split_target_features(config.name_of_target, xy_detector_fit)
    x_detector_val, _ = split_target_features(config.name_of_target, xy_detector_val)

    # train detectors and save them
    for detector_name in config.config_detector.detectors:
        # temporarily hacked tuning bool
        if detector_name.endswith("_untuned"):
            tuning = False
            detector_name = detector_name.removesuffix("_untuned")
        else:
            tuning = config.config_detector.tuning_bool

        if tuning:
            detector, threshold = tune_detector(
                detector_name,
                x_detector_fit,
                x_detector_val,
                true_validity_detector_val,
                config.config_detector,
            )

            # save detector and threshold
            save_path = os.path.join(config.experiment_folder, "detectors")
            loading_saving_aixtra.write_pkl(detector, f"{detector_name}_{tag}", save_path)
            loading_saving_aixtra.write_csv(
                threshold, f"{detector_name}_{tag}_threshold", save_path
            )
            loading_saving_aixtra.write_csv(
                x_detector_fit, f"{detector_name}_{tag}_x_fit", save_path
            )
            loading_saving_aixtra.write_csv(
                x_detector_val, f"{detector_name}_{tag}_x_val", save_path
            )
        else:
            # untuned detector with calculated nd_threshold
            # infer the actual fraction of outliers from the validation set
            actual_outlier_fraction = true_validity_detector_val.mean()
            detector, threshold = untuned_detector(
                detector_name, x_detector_fit, x_detector_val, actual_outlier_fraction
            )

            # save detector and threshold
            save_path = os.path.join(config.experiment_folder, "detectors")
            loading_saving_aixtra.write_pkl(
                detector, f"{detector_name}_{tag}_untuned", save_path
            )
            loading_saving_aixtra.write_csv(
                threshold, f"{detector_name}_{tag}_untuned_threshold", save_path
            )
            loading_saving_aixtra.write_csv(
                x_detector_fit, f"{detector_name}_{tag}_untuned_x_fit", save_path
            )
            loading_saving_aixtra.write_csv(
                x_detector_val, f"{detector_name}_{tag}_untuned_x_val", save_path
            )



    print(f"{__name__} executed")


if __name__ == "__main__":
    config = ExtrapolationExperimentConfig()
    exe(config)

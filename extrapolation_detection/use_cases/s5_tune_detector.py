import os

import pandas as pd

from extrapolation_detection.util import data_handling
from extrapolation_detection.use_cases.config.ed_experiment_config import ExtrapolationExperimentConfig
from extrapolation_detection.detector.tuning_mechanisms import tune_detector, untuned_detector
from core.util.data_handling import split_target_features

def exe_train_detector(config: ExtrapolationExperimentConfig):

    xy_training = data_handling.read_csv("xy_train", directory=config.experiment_folder)
    xy_validation = data_handling.read_csv("xy_val", directory=config.experiment_folder)
    xy_test = data_handling.read_csv("xy_test", directory=config.experiment_folder)
    xy_remaining = data_handling.read_csv("xy_remaining", directory=config.experiment_folder)

    true_validity_train = data_handling.read_csv("true_validity_train",
                                                 directory=config.experiment_folder).squeeze()
    true_validity_val = data_handling.read_csv("true_validity_val",
                                               directory=config.experiment_folder).squeeze()
    true_validity_test = data_handling.read_csv("true_validity_test",
                                                directory=config.experiment_folder).squeeze()
    true_validity_remaining = data_handling.read_csv("true_validity_remaining",
                                                     directory=config.experiment_folder).squeeze()

    xy_train_new, xy_val_new, true_validity_train_new, true_validity_val_new = data_handling.move_true_invalid_from_training_2_validation(xy_training, xy_validation, true_validity_train, true_validity_val)

    # handle different option for the validation set
    tag = "val"
    if config.detector_config.use_test_for_validation:
        xy_val_new = pd.concat([xy_val_new, xy_test])
        true_validity_val_new = pd.concat([true_validity_val_new, true_validity_test])
        tag = tag+"+test"
    if config.detector_config.use_remaining_for_validation:
        xy_val_new = pd.concat([xy_val_new, xy_remaining])
        true_validity_val_new = pd.concat([true_validity_val_new, true_validity_remaining])
        tag = tag+"+remaining"
    if config.detector_config.use_train_for_validation:
        xy_val_new = pd.concat([xy_val_new, xy_training])
        true_validity_val_new = pd.concat([true_validity_val_new, true_validity_train])
        tag = tag+"+train"
    if tag == "val+test+remaining+train":
        tag = "ideal"

    # delete target, as it is not needed for the detector
    x_train, _ = split_target_features(config.name_of_target, xy_train_new)
    x_val, _ = split_target_features(config.name_of_target, xy_val_new)

    # train detectors and save them
    for detector_name in config.detector_config.detectors:
        if config.detector_config.tuning_bool:
            detector, threshold = tune_detector(detector_name, x_train, x_val,
                           true_validity_val_new, config.detector_config)

            # save detector and threshold
            save_path = os.path.join(config.experiment_folder, "detectors")
            data_handling.write_pkl(detector, f"{detector_name}_{tag}", save_path)
            data_handling.write_csv(threshold, f"{detector_name}_{tag}_threshold", save_path)
        else:
            detector, threshold = untuned_detector(detector_name, x_train, x_val, config.true_outlier_fraction)

            # save detector and threshold
            save_path = os.path.join(config.experiment_folder, "detectors")
            data_handling.write_pkl(detector, f"{detector_name}_{tag}_untuned", save_path)
            data_handling.write_csv(threshold, f"{detector_name}_{tag}_untuned_threshold",
                                    save_path)


if __name__ == "__main__":
    config = ExtrapolationExperimentConfig()
    exe_train_detector(config)


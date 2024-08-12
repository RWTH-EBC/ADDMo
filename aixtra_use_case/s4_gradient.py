import os
import pandas as pd

from addmo.util.experiment_logger import ExperimentLogger
from aixtra.util import loading_saving_aixtra
from aixtra_use_case.config.ed_experiment_config import (
    ExtrapolationExperimentConfig,
)
from addmo.s3_model_tuning.models.abstract_model import AbstractMLModel
from aixtra.extrapolation_detection.n_D_extrapolation.gradients_analysis import calc_gradient
from aixtra.extrapolation_detection.n_D_extrapolation.gradients_analysis import classify_gradient

def exe(config: ExtrapolationExperimentConfig):
    """Calculates gradients for the model."""
    # load model
    regressor: AbstractMLModel = loading_saving_aixtra.load_regressor("regressor", directory=os.path.join(config.experiment_folder, "regressors"))

    # Load data
    x_train = loading_saving_aixtra.read_csv("x_train", directory=config.experiment_folder)
    x_val = loading_saving_aixtra.read_csv("x_val", directory=config.experiment_folder)
    x_test = loading_saving_aixtra.read_csv("x_test", directory=config.experiment_folder)
    x_remaining = loading_saving_aixtra.read_csv("x_remaining", directory=config.experiment_folder)
    x_grid = loading_saving_aixtra.read_csv("x_grid", directory=config.experiment_folder)

    # Calculate gradients
    gradients_train = calc_gradient(regressor, x_train)
    gradients_val = calc_gradient(regressor, x_val)
    gradients_test = calc_gradient(regressor, x_test)
    gradients_remaining = calc_gradient(regressor, x_remaining)
    gradients_grid = calc_gradient(regressor, x_grid)

    # Classify gradients
    gradients_clf_train = classify_gradient(gradients_train, config.gradient_zero_margin)
    gradients_clf_val = classify_gradient(gradients_val, config.gradient_zero_margin)
    gradients_clf_test = classify_gradient(gradients_test, config.gradient_zero_margin)
    gradients_clf_remaining = classify_gradient(gradients_remaining, config.gradient_zero_margin)
    gradients_clf_grid = classify_gradient(gradients_grid, config.gradient_zero_margin)

    # Save gradients to csv
    loading_saving_aixtra.write_csv(gradients_train, "gradients_train", directory=config.experiment_folder)
    loading_saving_aixtra.write_csv(gradients_val, "gradients_val", directory=config.experiment_folder)
    loading_saving_aixtra.write_csv(gradients_test, "gradients_test", directory=config.experiment_folder)
    loading_saving_aixtra.write_csv(gradients_remaining, "gradients_remaining", directory=config.experiment_folder)
    loading_saving_aixtra.write_csv(gradients_grid, "gradients_grid", directory=config.experiment_folder)

    # Save classified gradients to csv
    loading_saving_aixtra.write_csv(gradients_clf_train, "gradients_clf_train", directory=config.experiment_folder)
    loading_saving_aixtra.write_csv(gradients_clf_val, "gradients_clf_val", directory=config.experiment_folder)
    loading_saving_aixtra.write_csv(gradients_clf_test, "gradients_clf_test", directory=config.experiment_folder)
    loading_saving_aixtra.write_csv(gradients_clf_remaining, "gradients_clf_remaining", directory=config.experiment_folder)
    loading_saving_aixtra.write_csv(gradients_clf_grid, "gradients_clf_grid", directory=config.experiment_folder)

    print(f"{__name__} executed")

if __name__ == "__main__":
    config = ExtrapolationExperimentConfig()
    exe(config)
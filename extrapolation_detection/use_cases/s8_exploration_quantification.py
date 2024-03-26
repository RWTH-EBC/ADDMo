import os

import pandas as pd

from extrapolation_detection.util import data_handling
from extrapolation_detection.use_cases.config.ed_experiment_config import (
    ExtrapolationExperimentConfig,
)
from exploration_quantification.config.explo_quant_config import ExploQuantConfig
from extrapolation_detection.plotting import plot

from exploration_quantification import point_generator
from exploration_quantification.exploration_quantifier import ExplorationQuantifier


def exe_exploration_quantification(config: ExtrapolationExperimentConfig):
    # without target
    regressor_directory = os.path.join(config.experiment_folder, "regressors")
    x_regressor_fit = data_handling.read_csv(
        "x_regressor_fit", directory=regressor_directory
    )
    y_regressor_fit = data_handling.read_csv(
        "y_regressor_fit", directory=regressor_directory
    )

    # define bounds
    if config.explo_quant_config.exploration_bounds == "infer":
        bounds = point_generator.infer_meshgrid_bounds(x_regressor_fit)
    else:
        bounds = config.explo_quant_config.exploration_bounds

    # generate meshgrid
    grid_points = point_generator.generate_point_grid(
        x_regressor_fit, bounds, config.explo_quant_config.explo_grid_points_per_axis
    )

    save_path = os.path.join(config.experiment_folder, "explo_quant")
    data_handling.write_csv(grid_points, f"grid_points", save_path)

    for explo_detector_name in config.explo_quant_config.detectors:
        quantifier = ExplorationQuantifier(x_regressor_fit, grid_points, bounds)
        quantifier.train_exploration_classifier(explo_detector_name)
        quantifier.calc_labels()
        exploration_percentage = quantifier.calculate_exploration_percentages()

        plots_per_axes = quantifier.plot_scatter_extrapolation_share_2D(explo_detector_name)
        for i, plt in enumerate(plots_per_axes):
            plot.save_plot(
                plt,
                f"{explo_detector_name}_{i}",
                config.experiment_folder,
            )
            plot.show_plot(plt)

        # save

        data_handling.write_csv(quantifier.points_labeled, f"points_classified_{explo_detector_name}", save_path)
        data_handling.write_pkl(quantifier.explo_clf, f"explo_clf_{explo_detector_name}", save_path)
        data_handling.write_csv(
            exploration_percentage, f"exploration_percentage_{explo_detector_name}", save_path
        )


if __name__ == "__main__":
    config = ExtrapolationExperimentConfig()
    exe_exploration_quantification(config)

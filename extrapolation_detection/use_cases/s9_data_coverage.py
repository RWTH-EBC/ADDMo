import os

import matplotlib.pyplot as plt
import pandas as pd
from extrapolation_detection.util import loading_saving
from extrapolation_detection.use_cases.config.ed_experiment_config import (
    ExtrapolationExperimentConfig,
)

from extrapolation_detection.plotting import plot
from exploration_quantification import point_generator
from exploration_quantification import coverage_plotting


def exe(config: ExtrapolationExperimentConfig):
    # without target
    regressor_directory = os.path.join(config.experiment_folder, "regressors")
    xy_regressor_fit = loading_saving.read_csv(
        "xy_regressor_fit", directory=regressor_directory
    )

    # add predicted target (optional)
    y_pred = loading_saving.read_csv(
        "pred_regressor_fit", directory=regressor_directory
    )
    xy_regressor_fit["y_pred"] = y_pred
    config.config_explo_quant.exploration_bounds["y_pred"] = config.config_explo_quant.exploration_bounds[config.name_of_target]

    # define bounds
    bounds = point_generator.infer_or_forward_bounds(
        config.config_explo_quant.exploration_bounds, xy_regressor_fit
    )





    # plot
    plotly_parallel_coordinates_plt = (
        coverage_plotting.plot_dataset_parallel_coordinates_plotly(
            xy_regressor_fit, bounds, "Data coverage"
        )
    )
    scatter_matrix_plt = coverage_plotting.plot_dataset_distribution_kde(
        xy_regressor_fit, bounds, ""
    )

    # save plots
    save_path = os.path.join(config.experiment_folder)

    plot.save_plot(
        plotly_parallel_coordinates_plt, "plotly_parallel_coordinates", save_path
    )
    plot.save_plot(scatter_matrix_plt, "scatter_matrix", save_path)

    # plot.show_plot(plotly_parallel_coordinates_plt)
    # plot.show_plot(scatter_matrix_plt)

    print(f"{__name__} executed")


if __name__ == "__main__":
    config = ExtrapolationExperimentConfig()
    exe(config)

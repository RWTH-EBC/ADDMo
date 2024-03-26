import os

import pandas as pd

import extrapolation_detection.util.loading_saving
from extrapolation_detection.util import data_handling
from extrapolation_detection.use_cases.config.ed_experiment_config import (
    ExtrapolationExperimentConfig,
)
from exploration_quantification.config.explo_quant_config import ExploQuantConfig
from extrapolation_detection.plotting import plot

from exploration_quantification import point_generator
from exploration_quantification.exploration_quantifier import ExplorationQuantifier
from exploration_quantification import coverage_plotting


def exe_data_coverage(config: ExtrapolationExperimentConfig):
    # without target
    regressor_directory = os.path.join(config.experiment_folder, "regressors")
    x_regressor_fit = extrapolation_detection.util.loading_saving.read_csv(
        "x_regressor_fit", directory=regressor_directory
    )

    # define bounds
    if config.config_explo_quant.exploration_bounds == "infer":
        bounds = point_generator.infer_meshgrid_bounds(x_regressor_fit)
    else:
        bounds = config.config_explo_quant.exploration_bounds

    # plot
    plotly_parallel_coordinates_plt = coverage_plotting.plot_dataset_parallel_coordinates_plotly(
        x_regressor_fit, bounds, "hallo"
    )
    scatter_matric_plt = coverage_plotting.plot_dataset_distribution_kde(
        x_regressor_fit, bounds, "hallo"
    )
    pandas_parallel_coord = coverage_plotting.plot_dataset_parallel_coordinates(x_regressor_fit,
                                                                            "hallo")

    # save plots
    save_path = os.path.join(config.experiment_folder, "explo_quant")

    plot.save_plot(plotly_parallel_coordinates_plt, "plotly_parallel_coordinates",save_path)
    plot.save_plot(scatter_matric_plt, "scatter_matrix",save_path)
    plot.save_plot(pandas_parallel_coord, "pandas_parallel_coord", save_path)

    plot.show_plot(plotly_parallel_coordinates_plt)
    plot.show_plot(scatter_matric_plt)
    plot.show_plot(pandas_parallel_coord)


if __name__ == "__main__":
    config = ExtrapolationExperimentConfig()
    exe_data_coverage(config)

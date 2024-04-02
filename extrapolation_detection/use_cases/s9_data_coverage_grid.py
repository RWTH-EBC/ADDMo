import os

import matplotlib.pyplot as plt
import pandas as pd

from core.util.load_save import load_config_from_json

from extrapolation_detection.util import loading_saving
from extrapolation_detection.use_cases.config.ed_experiment_config import (
    ExtrapolationExperimentConfig,
)

from extrapolation_detection.plotting import plot
from exploration_quantification import point_generator
from exploration_quantification import coverage_plotting


def exe(config: ExtrapolationExperimentConfig):
    xy_grid = loading_saving.read_csv(
        "xy_grid", directory=config.experiment_folder
    )

    # add predicted target (optional)
    y_pred = loading_saving.read_csv(
        "errors_grid", directory=config.experiment_folder
    )["y_pred"]
    xy_grid["y_pred"] = y_pred

    # define bounds
    if config.config_explo_quant.exploration_bounds == "infer":
        bounds = point_generator.infer_meshgrid_bounds(xy_grid)
    else:
        bounds = config.config_explo_quant.exploration_bounds
    if "y_pred" in xy_grid.columns:
        bounds["y_pred"] = bounds[config.name_of_target]

    # plot
    plotly_parallel_coordinates_plt = (
        coverage_plotting.plot_dataset_parallel_coordinates_plotly(
            xy_grid, bounds, "Data coverage"
        )
    )

    # save plots
    save_path = os.path.join(config.experiment_folder)

    plot.save_plot(
        plotly_parallel_coordinates_plt, "plotly_parallel_coordinates_grid", save_path
    )

    plot.show_plot(plotly_parallel_coordinates_plt)

    print(f"{__name__} executed")


if __name__ == "__main__":
    config = ExtrapolationExperimentConfig()

    path = r"D:\04_GitRepos\addmo-extra\extrapolation_detection\use_cases\results\Boptest_TAir_mid_ODE_test1_supersmallANN\config.json"
    config = load_config_from_json(path, ExtrapolationExperimentConfig())
    exe(config)

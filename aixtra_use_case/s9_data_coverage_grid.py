import os

from addmo.util.load_save import load_config_from_json

from aixtra.util import loading_saving_aixtra
from aixtra_use_case.config.ed_experiment_config import (
    ExtrapolationExperimentConfig,
)

from aixtra.plotting import plot
from aixtra.exploration_quantification import point_generator, coverage_plotting


def exe(config: ExtrapolationExperimentConfig):
    xy_grid = loading_saving_aixtra.read_csv(
        "xy_grid", directory=config.experiment_folder
    )

    # add predicted target (optional)
    y_pred = loading_saving_aixtra.read_csv(
        "errors_grid", directory=config.experiment_folder
    )["y_pred"]
    xy_grid["y_pred"] = y_pred
    config.config_explo_quant.exploration_bounds["y_pred"] = config.config_explo_quant.exploration_bounds[config.name_of_target]

    # define bounds
    bounds = point_generator.infer_or_forward_bounds(
        config.config_explo_quant.exploration_bounds, xy_grid
    )

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

    # plot.show_plot(plotly_parallel_coordinates_plt)

    print(f"{__name__} executed")


if __name__ == "__main__":
    config = ExtrapolationExperimentConfig()

    path = r"/aixtra/extrapolation_detection\use_cases\results\Boptest_TAir_mid_ODE_test1_supersmallANN\config.json"
    config = load_config_from_json(path, ExtrapolationExperimentConfig())
    exe(config)

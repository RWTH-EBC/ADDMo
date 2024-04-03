import os

from core.util.experiment_logger import ExperimentLogger
from core.util.load_save import load_config_from_json

from extrapolation_detection.util import loading_saving
from extrapolation_detection.use_cases.config.ed_experiment_config import (
    ExtrapolationExperimentConfig,
)

from extrapolation_detection.plotting import plot

from exploration_quantification import point_generator
from exploration_quantification.exploration_quantifier import GridOccupancy
from exploration_quantification.coverage_plotting import (
    plot_grid_cells_average_coverage_per_2D,
)


def exe(config: ExtrapolationExperimentConfig):
    # without target
    regressor_directory = os.path.join(config.experiment_folder, "regressors")
    x_regressor_fit = loading_saving.read_csv(
        "x_regressor_fit", directory=regressor_directory
    )

    # define bounds
    if config.config_explo_quant.exploration_bounds == "infer":
        bounds = point_generator.infer_meshgrid_bounds(x_regressor_fit)
    else:
        bounds = config.config_explo_quant.exploration_bounds

    grid_occupancy = GridOccupancy(config.config_explo_quant.explo_grid_points_per_axis)
    grid_occupancy.train(x_regressor_fit, bounds)
    coverage = grid_occupancy.calculate_coverage()

    plots_per_axes = plot_grid_cells_average_coverage_per_2D(
        coverage_grid=grid_occupancy.occupancy_grid,
        boundaries=bounds,
        variable_names=x_regressor_fit.columns,
        title_header=f"Grid cells = {config.config_explo_quant.explo_grid_points_per_axis}; "
                     f"Coverage ="
                     f" {coverage.loc['Inside']} %",
    )

    for i, plt in enumerate(plots_per_axes):
        plot.save_plot(
            plt,
            f"coverage_grid_occupancy_{i}",
            config.experiment_folder,
        )
        plot.show_plot(plt)

    # log coverage
    ExperimentLogger.log({f"coverage_grid_occupancy": coverage.loc["Inside"]})

    print(f"{__name__} executed")


if __name__ == "__main__":
    config = ExtrapolationExperimentConfig()

    path = r"D:\04_GitRepos\addmo-extra\extrapolation_detection\use_cases\results\Boptest_TAir_mid_ODE_test1_supersmallANN\config.json"
    config = load_config_from_json(path, ExtrapolationExperimentConfig())
    exe(config)

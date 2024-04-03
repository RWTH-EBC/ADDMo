import os
import pandas as pd

from core.util.experiment_logger import ExperimentLogger
from core.s3_model_tuning.models.abstract_model import AbstractMLModel
from extrapolation_detection.n_D_extrapolation.score_regressor_per_data_point import score_per_sample
from extrapolation_detection.n_D_extrapolation import true_validity_domain
from extrapolation_detection.util import loading_saving
from extrapolation_detection.use_cases.config.ed_experiment_config import (
    ExtrapolationExperimentConfig,
)
from extrapolation_detection.system_simulations import system_simulations
from extrapolation_detection.plotting import plot

from exploration_quantification import point_generator
from exploration_quantification.exploration_quantifier import ExplorationQuantifier
from exploration_quantification.coverage_plotting import (
    plot_scatter_average_coverage_per_2D,
)


def exe(config: ExtrapolationExperimentConfig):

    grid_path = os.path.join(config.experiment_folder, "explo_quant")
    x_grid = loading_saving.read_csv(f"grid_points", grid_path)

    regressor: AbstractMLModel = loading_saving.read_pkl(
        "regressor", directory=os.path.join(config.experiment_folder, "regressors")
    )

    # generate y values for the grid
    y_grid = system_simulations.simulate(x_grid, config.system_simulation)
    y_grid.name = config.name_of_target

    errors_grid = score_per_sample(
        regressor, x_grid, y_grid, metric=config.true_outlier_threshold_error_metric
    )

    true_validity_threshold = loading_saving.read_csv(
        "true_validity_threshold", directory=config.experiment_folder
    ).iloc[0, 0]
    true_validity_grid = true_validity_domain.classify_errors_2_true_validity(
        errors_grid["error"], true_validity_threshold
    )

    quantifier = ExplorationQuantifier()
    y_grid = true_validity_grid
    quantifier.labels_grid = true_validity_grid
    quantifier.x_grid = x_grid
    coverage = quantifier.calculate_coverage()

    plots_per_axes = plot_scatter_average_coverage_per_2D(
        x_grid=x_grid,
        y_grid=y_grid,
        title_header=f"true validity\n"
                     f"Coverage = {coverage.loc['Inside']:.2f} %",
    )

    for i, plt in enumerate(plots_per_axes):
        plot.save_plot(
            plt,
            f"coverage_true validity_{i}",
            config.experiment_folder,
        )
        plot.show_plot(plt)

    # save
    save_path = os.path.join(config.experiment_folder, "explo_quant")
    loading_saving.write_csv(
        quantifier.labels_grid,
        f"labels_grid_true validity",
        save_path,
    )
    loading_saving.write_csv(
        coverage,
        f"coverage_percentage_true validity",
        save_path,
    )

    # log coverage
    ExperimentLogger.log({f"coverage_true_validity": coverage.loc["Inside"]})

    print(f"{__name__} executed")


if __name__ == "__main__":
    config = ExtrapolationExperimentConfig()
    exe(config)

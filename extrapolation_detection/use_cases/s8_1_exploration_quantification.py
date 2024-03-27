import os

import extrapolation_detection.util.loading_saving
from extrapolation_detection.use_cases.config.ed_experiment_config import (
    ExtrapolationExperimentConfig,
)

from extrapolation_detection.plotting import plot

from exploration_quantification import point_generator
from exploration_quantification.exploration_quantifier import ExplorationQuantifier
from exploration_quantification.coverage_plotting import (
    plot_scatter_average_coverage_per_2D,
)


def exe_exploration_quantification(config: ExtrapolationExperimentConfig):
    # without target
    regressor_directory = os.path.join(config.experiment_folder, "regressors")
    x_regressor_fit = extrapolation_detection.util.loading_saving.read_csv(
        "x_regressor_fit", directory=regressor_directory
    )
    y_regressor_fit = extrapolation_detection.util.loading_saving.read_csv(
        "y_regressor_fit", directory=regressor_directory
    )

    # define bounds
    if config.config_explo_quant.exploration_bounds == "infer":
        bounds = point_generator.infer_meshgrid_bounds(x_regressor_fit)
    else:
        bounds = config.config_explo_quant.exploration_bounds

    # generate meshgrid
    x_grid = point_generator.generate_point_grid(
        x_regressor_fit, bounds, config.config_explo_quant.explo_grid_points_per_axis
    )

    save_path = os.path.join(config.experiment_folder, "explo_quant")
    extrapolation_detection.util.loading_saving.write_csv(
        x_grid, f"grid_points", save_path
    )

    for explo_detector_name in config.config_explo_quant.detectors:
        quantifier = ExplorationQuantifier(x_regressor_fit, x_grid, bounds)
        quantifier.train_exploration_classifier(explo_detector_name)
        y_grid = quantifier.calc_labels()
        exploration_percentage = quantifier.calculate_coverage()

        plots_per_axes = plot_scatter_average_coverage_per_2D(
            x_grid=x_grid,
            y_grid=y_grid,
            title_header=f"{explo_detector_name}"
                         f" with coverage = {exploration_percentage.loc['Inside']} %",
        )
        for i, plt in enumerate(plots_per_axes):
            plot.save_plot(
                plt,
                f"{explo_detector_name}_{i}",
                config.experiment_folder,
            )
            plot.show_plot(plt)

        # save
        extrapolation_detection.util.loading_saving.write_csv(
            quantifier.points_labeled,
            f"points_classified_{explo_detector_name}",
            save_path,
        )
        extrapolation_detection.util.loading_saving.write_pkl(
            quantifier.explo_clf, f"explo_clf_{explo_detector_name}", save_path
        )
        extrapolation_detection.util.loading_saving.write_csv(
            exploration_percentage,
            f"exploration_percentage_{explo_detector_name}",
            save_path,
        )

    print(f"{__name__} executed")


if __name__ == "__main__":
    config = ExtrapolationExperimentConfig()
    exe_exploration_quantification(config)

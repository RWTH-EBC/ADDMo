import os

from core.util.experiment_logger import ExperimentLogger

from extrapolation_detection.util import loading_saving
from extrapolation_detection.use_cases.config.ed_experiment_config import (
    ExtrapolationExperimentConfig,
)

from extrapolation_detection.plotting import plot

from exploration_quantification import point_generator
from exploration_quantification.exploration_quantifier import ExplorationQuantifier
from exploration_quantification.coverage_plotting import (
    plot_scatter_average_coverage_per_2D,
)


def exe(config: ExtrapolationExperimentConfig):
    # without target
    regressor_directory = os.path.join(config.experiment_folder, "regressors")
    x_regressor_fit = loading_saving.read_csv(
        "x_regressor_fit", directory=regressor_directory
    )
    y_regressor_fit = loading_saving.read_csv(
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
    loading_saving.write_csv(
        x_grid, f"grid_points", save_path
    )

    for explo_detector_name in config.config_explo_quant.detectors:
        quantifier = ExplorationQuantifier()
        quantifier.train_exploration_classifier(x_regressor_fit, explo_detector_name)
        y_grid = quantifier.calc_labels(x_grid)
        coverage = quantifier.calculate_coverage()

        plots_per_axes = plot_scatter_average_coverage_per_2D(
            x_grid=x_grid,
            y_grid=y_grid,
            title_header=f"{explo_detector_name}\n"
                         f"Coverage = {coverage.loc['Inside']:.2f} %",
        )
        save_path = os.path.join(config.experiment_folder, "explo_quant")
        for i, plt in enumerate(plots_per_axes):
            plot.save_plot(
                plt,
                f"coverage_{explo_detector_name}_{i}",
                config.experiment_folder,
            )
            plot.show_plot(plt)

        # save
        loading_saving.write_csv(
            quantifier.labels_grid,
            f"labels_grid_{explo_detector_name}",
            save_path,
        )
        loading_saving.write_pkl(
            quantifier.explo_clf, f"explo_clf_{explo_detector_name}", save_path
        )
        loading_saving.write_csv(
            coverage,
            f"coverage_percentage_{explo_detector_name}",
            save_path,
        )

        # log coverage
        ExperimentLogger.log({f"coverage_{explo_detector_name}": coverage.loc["Inside"]})
        ExperimentLogger.log({"bounds": bounds})

    print(f"{__name__} executed")


if __name__ == "__main__":
    config = ExtrapolationExperimentConfig()
    exe(config)

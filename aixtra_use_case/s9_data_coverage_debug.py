import os

from aixtra.util import loading_saving_aixtra
from aixtra_use_case.config.ed_experiment_config import (
    ExtrapolationExperimentConfig,
)

from aixtra.plotting import plot
from aixtra.exploration_quantification import point_generator, coverage_plotting


def exe(config: ExtrapolationExperimentConfig):
    # without target
    regressor_directory = os.path.join(config.experiment_folder, "detectors")
    xy_regressor_fit = loading_saving_aixtra.read_csv(
        "KNN_test+fit_x_fit", directory=regressor_directory
    )

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

    # save model_plots
    save_path = os.path.join(config.experiment_folder)

    plot.save_plot(
        plotly_parallel_coordinates_plt, "plotly_parallel_coordinates_detector_fit", save_path
    )
    plot.save_plot(scatter_matrix_plt, "scatter_matrix_detector_fit", save_path)

    # plot.show_plot(plotly_parallel_coordinates_plt)
    # plot.show_plot(scatter_matrix_plt)

    print(f"{__name__} executed")


if __name__ == "__main__":
    config = ExtrapolationExperimentConfig()
    exe(config)

import os

from aixtra.util import loading_saving_aixtra
from aixtra_use_case.config.ed_experiment_config import (
    ExtrapolationExperimentConfig,
)

from aixtra.plotting import plot
from aixtra.exploration_quantification import point_generator, coverage_plotting


def exe(config: ExtrapolationExperimentConfig):
    # without target
    regressor_directory = os.path.join(config.experiment_folder, "regressors")
    xy_regressor_fit = loading_saving_aixtra.read_csv(
        "xy_regressor_fit", directory=regressor_directory
    )

    # add predicted target (optional)
    y_pred = loading_saving_aixtra.read_csv(
        "pred_regressor_fit", directory=regressor_directory
    )
    xy_regressor_fit["y_pred"] = y_pred
    config.config_explo_quant.exploration_bounds["y_pred"] = config.config_explo_quant.exploration_bounds[config.name_of_target]

    # define bounds
    bounds = point_generator.infer_or_forward_bounds(
        config.config_explo_quant.exploration_bounds, xy_regressor_fit
    )

    # for nicer appearence make sure that the control variable 3rd last column
    control_var = config.var4gradient
    cols = xy_regressor_fit.columns.tolist()
    cols.remove(control_var)
    cols.insert(-2, control_var)
    xy_regressor_fit = xy_regressor_fit[cols]



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

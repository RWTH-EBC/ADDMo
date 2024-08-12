import os

from addmo.util.experiment_logger import ExperimentLogger
from addmo.s3_model_tuning.models.abstract_model import AbstractMLModel
from aixtra.util.loading_saving_aixtra import load_regressor
from aixtra.extrapolation_detection.n_D_extrapolation.gradients_analysis import calc_gradient
from aixtra.extrapolation_detection.n_D_extrapolation.gradients_analysis import classify_gradient
from aixtra.extrapolation_detection.n_D_extrapolation.gradients_analysis import calc_gradient_coverage
from aixtra.util import loading_saving_aixtra
from aixtra_use_case.config.ed_experiment_config import (
    ExtrapolationExperimentConfig,
)
from aixtra.plotting import plot

from aixtra.exploration_quantification.exploration_quantifier import ExplorationQuantifier
from aixtra.exploration_quantification.coverage_plotting import (
    plot_scatter_average_coverage_per_2D,
)


def exe(config: ExtrapolationExperimentConfig):

    grid_path = os.path.join(config.experiment_folder, "explo_quant")
    x_grid = loading_saving_aixtra.read_csv(f"grid_points", grid_path)

    regressor: AbstractMLModel = load_regressor(
        "regressor", directory=os.path.join(config.experiment_folder, "regressors"))


    # calc gradients
    gradients_grid = calc_gradient(regressor, x_grid)
    gradients_classified_grid = classify_gradient(gradients_grid, config.gradient_zero_margin)
    gradients_classified_grid_var4gradient = gradients_classified_grid[config.var4gradient]

    coverage = calc_gradient_coverage(gradients_classified_grid_var4gradient)

    # Todo: Make all code (true validity, detectors, etc.) have inside converage as 1 und outside as 0 and not the other way around
    wrong_gradient = gradients_classified_grid_var4gradient != config.correct_gradient

    gradient_str = "positive" if config.correct_gradient == 1 else "negative" if config.correct_gradient == -1 else "zero"

    plots_per_axes = plot_scatter_average_coverage_per_2D(
        x_grid=x_grid,
        y_grid=wrong_gradient,
        title_header=f"Desired gradient: {gradient_str}\n"
                     f"pos: {coverage.loc['Increasing']:.2f} %"
                     f" | neg: {coverage.loc['Decreasing']:.2f} %"
                     f" | zero: {coverage.loc['Constant']:.2f} %",
    )

    for i, plt in enumerate(plots_per_axes):
        plot.save_plot(
            plt,
            f"coverage_gradient_{i}",
            config.experiment_folder,
        )
        # plot.show_plot(plt)

    # save
    save_path = os.path.join(config.experiment_folder, "explo_quant")
    loading_saving_aixtra.write_csv(
        gradients_grid,
        f"gradients_grid",
        save_path,
    )
    loading_saving_aixtra.write_csv(
        gradients_classified_grid,
        f"gradients_clf_grid",
        save_path,
    )
    loading_saving_aixtra.write_csv(
        coverage,
        f"coverage_gradients",
        save_path,
    )

    # log coverage
    ExperimentLogger.log({f"coverage_gradient_pos": coverage.loc["Increasing"],
                          f"coverage_gradient_neg": coverage.loc["Decreasing"],
                          f"coverage_gradient_zero": coverage.loc["Constant"]})

    print(f"{__name__} executed")


if __name__ == "__main__":
    config = ExtrapolationExperimentConfig()
    exe(config)

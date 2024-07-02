import os

from addmo.util.experiment_logger import ExperimentLogger

from aixtra.util import loading_saving_aixtra
from aixtra_use_case.config.ed_experiment_config import (
    ExtrapolationExperimentConfig,
)

from aixtra.plotting import plot

from aixtra.exploration_quantification import point_generator
from aixtra.exploration_quantification.exploration_quantifier import ExplorationQuantifier
from aixtra.exploration_quantification.coverage_plotting import (
    plot_scatter_average_coverage_per_2D,
)


def exe(config: ExtrapolationExperimentConfig):
    # without target
    regressor_directory = os.path.join(config.experiment_folder, "regressors")
    x_regressor_fit = loading_saving_aixtra.read_csv(
        "x_regressor_fit", directory=regressor_directory
    )

    bounds = point_generator.infer_or_forward_bounds(
        config.config_explo_quant.exploration_bounds, x_regressor_fit
    )

    # generate meshgrid
    x_grid = point_generator.generate_point_grid(
        x_regressor_fit, bounds, config.config_explo_quant.explo_grid_points_per_axis
    )

    save_path = os.path.join(config.experiment_folder, "explo_quant")
    loading_saving_aixtra.write_csv(x_grid, f"grid_points", save_path)

    ExperimentLogger.log({"bounds": bounds})

    print(f"{__name__} executed")


if __name__ == "__main__":
    config = ExtrapolationExperimentConfig()
    exe(config)

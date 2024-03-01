import pandas as pd

from extrapolation_detection.machine_learning_util import data_handling
from extrapolation_detection.new_use_cases.ed_experiment_config import (
    ExtrapolationExperimentConfig,
)

from extrapolation_detection.machine_learning_util import data_handling as dh
from core.util.data_handling import split_target_features
from core.exploration_quantification.exploration_quantification import ArtificialPointGenerator


def exe_split_data(config: ExtrapolationExperimentConfig):
    # Load data
    xy_tot = data_handling.read_csv(config.simulation_data_name, directory="data", index_col=False)

    (
        xy_training,
        xy_validation,
        xy_test,
        xy_remaining,
    ) = data_handling.split_simulation_data(
        xy_tot,
        config.train_val_test,
        config.val_fraction,
        config.test_fraction,
        config.shuffle,
    )

    # generate meshgrid
    grid_generator = ArtificialPointGenerator()
    x_tot, _ = split_target_features(config.name_of_target, xy_tot)
    bounds = grid_generator.infer_meshgrid_bounds(x_tot)
    x_grid = grid_generator.generate_point_grid(x_tot, bounds, config.grid_points_per_axis)

    # simulate true values for the grid via the system simulation
    if config.system_simulation == "carnot":
        from extrapolation_detection.system_simulations.carnot_model import carnot_model
        system_simulation = carnot_model

    y_grid = x_grid.apply(lambda row: system_simulation(*row), axis=1)
    y_grid.name = config.name_of_target
    xy_grid = pd.concat([x_grid, y_grid], axis=1)

    # save to csv
    data_handling.write_csv(xy_training, "xy_train", directory=config.experiment_name)
    data_handling.write_csv(xy_validation, "xy_val", directory=config.experiment_name)
    data_handling.write_csv(xy_test, "xy_test", directory=config.experiment_name)
    data_handling.write_csv(xy_remaining, "xy_remaining", directory=config.experiment_name)
    data_handling.write_csv(xy_grid, "xy_grid", directory=config.experiment_name)

    # additionally save the features and targets separately
    x_train, y_train = split_target_features(config.name_of_target, xy_training)
    x_val, y_val = split_target_features(config.name_of_target, xy_validation)
    x_test, y_test = split_target_features(config.name_of_target, xy_test)
    x_remaining, y_remaining = split_target_features(config.name_of_target, xy_remaining)
    x_grid, y_grid = split_target_features(config.name_of_target, xy_grid)

    data_handling.write_csv(x_train, "x_train", directory=config.experiment_name)
    data_handling.write_csv(y_train, "y_train", directory=config.experiment_name)
    data_handling.write_csv(x_val, "x_val", directory=config.experiment_name)
    data_handling.write_csv(y_val, "y_val", directory=config.experiment_name)
    data_handling.write_csv(x_test, "x_test", directory=config.experiment_name)
    data_handling.write_csv(y_test, "y_test", directory=config.experiment_name)
    data_handling.write_csv(x_remaining, "x_remaining", directory=config.experiment_name)
    data_handling.write_csv(y_remaining, "y_remaining", directory=config.experiment_name)
    data_handling.write_csv(x_grid, "x_grid", directory=config.experiment_name)
    data_handling.write_csv(y_grid, "y_grid", directory=config.experiment_name)

if __name__ == "__main__":
    config = ExtrapolationExperimentConfig()
    exe_split_data(config)

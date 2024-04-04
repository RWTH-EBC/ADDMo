import os.path

import pandas as pd

from core.util.definitions import ed_use_case_dir
from extrapolation_detection.util import loading_saving
from extrapolation_detection.util import data_handling
from extrapolation_detection.use_cases.config.ed_experiment_config import (
    ExtrapolationExperimentConfig,
)
from extrapolation_detection.system_simulations import system_simulations

from core.util.data_handling import split_target_features
from exploration_quantification import point_generator


def exe(config: ExtrapolationExperimentConfig):
    # Load data
    xy_tot = loading_saving.read_csv(
        config.simulation_data_name, directory=os.path.join(ed_use_case_dir(), "data"), index_col=False
    )

    train_val_test_indices = range(
        config.train_val_test_period[0], config.train_val_test_period[1]
    )

    (
        xy_training,
        xy_validation,
        xy_test,
        xy_remaining,
    ) = data_handling.split_simulation_data(
        xy_tot,
        train_val_test_indices,
        config.val_fraction,
        config.test_fraction,
        config.shuffle,
    )

    if config.system_simulation is None:
        # no system simulation available, just give it some "dummy" values for the program to work
        xy_grid = xy_tot.iloc[:5]
    else:
        # generate meshgrid
        x_tot, _ = split_target_features(config.name_of_target, xy_tot)
        bounds = point_generator.infer_or_forward_bounds(
            config.config_explo_quant.exploration_bounds, x_tot
        )
        x_grid = point_generator.generate_point_grid(
            x_tot, bounds, config.grid_points_per_axis
        )

        # generate y values for the grid
        y_grid = system_simulations.simulate(x_grid, config.system_simulation)
        y_grid.name = config.name_of_target
        xy_grid = pd.concat([x_grid, y_grid], axis=1)

    # save to csv
    loading_saving.write_csv(
        xy_training, "xy_train", directory=config.experiment_folder
    )
    loading_saving.write_csv(
        xy_validation, "xy_val", directory=config.experiment_folder
    )
    loading_saving.write_csv(xy_test, "xy_test", directory=config.experiment_folder)
    loading_saving.write_csv(
        xy_remaining, "xy_remaining", directory=config.experiment_folder
    )
    loading_saving.write_csv(xy_grid, "xy_grid", directory=config.experiment_folder)

    # additionally save the features and targets separately
    x_train, y_train = split_target_features(config.name_of_target, xy_training)
    x_val, y_val = split_target_features(config.name_of_target, xy_validation)
    x_test, y_test = split_target_features(config.name_of_target, xy_test)
    x_remaining, y_remaining = split_target_features(
        config.name_of_target, xy_remaining
    )
    x_grid, y_grid = split_target_features(config.name_of_target, xy_grid)

    loading_saving.write_csv(x_train, "x_train", directory=config.experiment_folder)
    loading_saving.write_csv(y_train, "y_train", directory=config.experiment_folder)
    loading_saving.write_csv(x_val, "x_val", directory=config.experiment_folder)
    loading_saving.write_csv(y_val, "y_val", directory=config.experiment_folder)
    loading_saving.write_csv(x_test, "x_test", directory=config.experiment_folder)
    loading_saving.write_csv(y_test, "y_test", directory=config.experiment_folder)
    loading_saving.write_csv(
        x_remaining, "x_remaining", directory=config.experiment_folder
    )
    loading_saving.write_csv(
        y_remaining, "y_remaining", directory=config.experiment_folder
    )
    loading_saving.write_csv(x_grid, "x_grid", directory=config.experiment_folder)
    loading_saving.write_csv(y_grid, "y_grid", directory=config.experiment_folder)

    print(f"{__name__} executed")


if __name__ == "__main__":
    config = ExtrapolationExperimentConfig()
    exe(config)

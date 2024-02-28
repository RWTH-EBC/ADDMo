experiment_name = "Carnot_Test"

from extrapolation_detection.machine_learning_util import data_handling
from extrapolation_detection.new_use_cases.ed_experiment_config import (
    ExtrapolationExperimentConfig,
)


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

    # save to csv
    data_handling.write_csv(xy_training, "xy_train", directory=experiment_name)
    data_handling.write_csv(xy_validation, "xy_val", directory=experiment_name)
    data_handling.write_csv(xy_test, "xy_test", directory=experiment_name)
    data_handling.write_csv(xy_remaining, "xy_remaining", directory=experiment_name)


if __name__ == "__main__":
    config = ExtrapolationExperimentConfig()
    exe_split_data(config)

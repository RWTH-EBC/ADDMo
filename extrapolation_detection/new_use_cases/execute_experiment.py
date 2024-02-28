from extrapolation_detection.machine_learning_util import data_handling
from extrapolation_detection.new_use_cases.ed_experiment_config import ExtrapolationExperimentConfig



config = ExtrapolationExperimentConfig()

# Load data
xy_tot = data_handling.load_csv(config.simulation_data_name, path="data")

data_handling.split_simulation_data(
        xy_tot,
        config.train_val_test,
        val_fraction=0.1,
        test_fraction=0.1,
        shuffle=True,
    )
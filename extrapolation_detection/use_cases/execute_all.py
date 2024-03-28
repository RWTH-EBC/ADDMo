import os

from core.util.experiment_logger import ExperimentLogger
from core.util.experiment_logger import LocalLogger
from core.util.experiment_logger import WandbLogger
from core.util.definitions import root_dir
from core.util.load_save import load_config_from_json
from core.util.definitions import results_dir_extrapolation_experiment
from core.s3_model_tuning.config.model_tuning_config import ModelTunerConfig

from extrapolation_detection.use_cases.config.ed_experiment_config import (
    ExtrapolationExperimentConfig,
)
from extrapolation_detection.use_cases import (
    s1_split_data,
    s2_tune_ml_regressor,
    s3_regressor_error_calculation,
    s4_true_validity_domain,
    s5_tune_detector,
    s6_detector_score_calculation,
    s7_2_plotting,
    s8_1_exploration_quantification,
    s8_2_exploration_quantification_grid_occupancy,
    s8_3_exploration_quantification_extra,
    s9_data_coverage,
)

# configure config
config = ExtrapolationExperimentConfig()
config.simulation_data_name = "Boptest_TAir_mid_ODE"
config.experiment_name = "Boptest_TAir_mid_ODE_small_ANN"
config.name_of_target = "delta_reaTZon_y"
config.train_val_test_period = (0, 1488)
config.shuffle = False
# config.grid_points_per_axis = 3
config.system_simulation = None

config.config_explo_quant.explo_grid_points_per_axis = 10

# Load the config from the json file
path_to_config = os.path.join(
    root_dir(), "core", "s3_model_tuning", "config", "model_tuner_config_no_tuning.json"
)
config.config_model_tuning = load_config_from_json(path_to_config, ModelTunerConfig)

config.config_model_tuning.hyperparameter_tuning_kwargs = {
    "hyperparameter_set": {
        "hidden_layer_sizes": [100, 100],
        "activation": "relu",
        "max_iter": 2000,
    }
}

config.config_detector.detectors = ["KNN", "KDE", "GP", "OCSVM", "IF"]

# Configure the logger
result_folder = results_dir_extrapolation_experiment(config)
LocalLogger.directory = result_folder
LocalLogger.active = True

# Run scripts
# ExperimentLogger.start_experiment(config=config)  # log config
# s1_split_data.exe_split_data(config)
# s2_tune_ml_regressor.exe_tune_regressor(config)
s3_regressor_error_calculation.exe_regressor_error_calculation(config)
# s4_true_validity_domain.exe_true_validity_domain(config)
# s5_tune_detector.exe_train_detector(config)
# s6_detector_score_calculation.exe_detector_score_calculation(config)
# s8_1_exploration_quantification.exe_exploration_quantification(config)
# s8_2_exploration_quantification_grid_occupancy.exe_exploration_quantification_grid_occupancy(
#     config
# )
# s8_3_exploration_quantification_extra.exe_exploration_quantification_extra(config)
# s9_data_coverage.exe_data_coverage(config)


# s7_2_plotting.exe_plot_2D_all(config)

import os

from core.util.experiment_logger import ExperimentLogger
from core.util.experiment_logger import LocalLogger
from core.util.experiment_logger import WandbLogger
from core.util.definitions import root_dir
from core.util.load_save import load_config_from_json
from core.util.load_save import save_config_to_json
from core.util.definitions import results_dir_extrapolation_experiment

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
    s9_data_coverage,
)

# load config
config = ExtrapolationExperimentConfig()
config.simulation_data_name = "Boptest_TAir_mid_reduced"
config.experiment_name = "test_Boptest_TAir_mid_reduced"
config.name_of_target = "delta_reaTZon_y"
# config.grid_points_per_axis = 3
config.system_simulation = None

config.config_explo_quant.explo_grid_points_per_axis = 10


result_folder = results_dir_extrapolation_experiment(config)

# Configure the logger
LocalLogger.directory = result_folder
LocalLogger.active = True

# Initialize logging
ExperimentLogger.start_experiment(config=config)

s1_split_data.exe_split_data(config)
s2_tune_ml_regressor.exe_tune_regressor(config)
s3_regressor_error_calculation.exe_regressor_error_calculation(config)
s4_true_validity_domain.exe_true_validity_domain(config)
s5_tune_detector.exe_train_detector(config)
s6_detector_score_calculation.exe_detector_score_calculation(config)
# s7_2_plotting.exe_plot_2D_all(config)
# s8_1_exploration_quantification.exe_exploration_quantification(config)
s8_2_exploration_quantification_grid_occupancy.exe_exploration_quantification_grid_occupancy(config)
s9_data_coverage.exe_data_coverage(config)
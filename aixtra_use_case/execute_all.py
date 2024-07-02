import os

from addmo.util.experiment_logger import ExperimentLogger
from addmo.util.experiment_logger import LocalLogger
from addmo.util.experiment_logger import WandbLogger
from addmo.util.definitions import create_or_clean_directory

from aixtra_use_case.config.ed_experiment_config import (
    ExtrapolationExperimentConfig,
)
from aixtra_use_case import s2_tune_ml_regressor, s4_true_validity_domain, \
    s8_1_coverage_convex_hull, s3_regressor_error_calculation, s8_3_coverage_tuned_ND, \
    s5_tune_detector, s1_split_data, s7_2_plotting, s8_4_coverage_true_validity, \
    s6_detector_score_calculation, s8_2_coverage_grid_occupancy, s9_data_coverage_grid, \
    s9_data_coverage, s8_0_generate_grid

# configure config
config = ExtrapolationExperimentConfig()
config.simulation_data_name = "Boptest_TAir_mid_ODE_noise_m0_std0.01"
config.experiment_name = "RISHIKA_TEST2"
config.name_of_target = "delta_reaTZon_y"
config.train_val_test_period = (0, 1488)
config.shuffle = False
config.grid_points_per_axis = 10
config.system_simulation = "BopTest_TAir_ODE"
config.true_outlier_threshold = 0.1

config.config_explo_quant.exploration_bounds = {
    "TDryBul": (263.15, 303.15),
    "HDirNor": (0, 1000),
    "oveHeaPumY_u": (0, 1),
    "reaTZon_y": (290.15, 300.15),
    "delta_reaTZon_y": (-0.5, 0.5),
}
# config= config_blueprints.no_tuning_config()
config.config_model_tuning.models = ["SciKerasSequential"]
config.config_model_tuning.hyperparameter_tuning_type = "OptunaTuner"
config.config_model_tuning.hyperparameter_tuning_kwargs = {"n_trials": 10}
config.config_model_tuning.validation_score_metric = "neg_root_mean_squared_error"

config.config_detector.detectors = ["KNN", "GP", "OCSVM"]

# Configure the logger
create_or_clean_directory(config.experiment_folder)
LocalLogger.directory = os.path.join(config.experiment_folder, "local_logger")
LocalLogger.active = True
WandbLogger.project = f"ED_{config.simulation_data_name}"
WandbLogger.directory = config.experiment_folder
WandbLogger.active = False


# Run scripts
ExperimentLogger.start_experiment(config=config)  # log config
s1_split_data.exe(config)
s2_tune_ml_regressor.exe(config)
s3_regressor_error_calculation.exe(config)
s4_true_validity_domain.exe(config)
# s5_tune_detector.exe(config)
# s6_detector_score_calculation.exe(config)
s8_0_generate_grid.exe(config)
# s8_1_coverage_convex_hull.exe(config)
# s8_2_coverage_grid_occupancy.exe(config)
# s8_3_coverage_tuned_ND.exe(config)
s8_4_coverage_true_validity.exe(config)
# s9_data_coverage.exe(config)
# s9_data_coverage_grid.exe(config)
# # #
# s7_2_plotting.exe_plot_2D_all(config)
# s7_2_plotting.exe_plot_2D_detector(config)

# from extrapolation_detection.aixtra_use_case import s9_data_coverage_debug
# s9_data_coverage_debug.exe(config)

ExperimentLogger.finish_experiment()

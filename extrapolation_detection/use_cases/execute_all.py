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
    s8_1_coverage_convex_hull,
    s8_2_coverage_grid_occupancy,
    s8_3_coverage_tuned_ND,
    s8_4_coverage_true_validity,
    s9_data_coverage,
    s9_data_coverage_grid,
)

# configure config
config = ExtrapolationExperimentConfig()
# config.simulation_data_name = "Boptest_TAir_mid_ODE"
config.experiment_name = "wandb_test_4"
# config.name_of_target = "delta_reaTZon_y"
# config.train_val_test_period = (0, 1488)
# config.shuffle = False
# config.grid_points_per_axis = 20
# config.system_simulation = "BopTest_TAir_ODE" # "carnot
# config.true_outlier_threshold = 0.111819189397924
#
# config.config_explo_quant.explo_grid_points_per_axis = 20
# # config.config_explo_quant.exploration_bounds = {
# #     "$T_{umg}$ in °C": (-7.5, 20),
# #     "$P_{el}$ in kW": (0, 4.5),
# #     "$\dot{Q}_{heiz}$ in kW": (0, 35),
# # }
# # #
# # # # Load the config from the json file
# path_to_config = os.path.join(
#     root_dir(), "core", "s3_model_tuning", "config", "model_tuner_config_no_tuning.json"
# )
# config.config_model_tuning = load_config_from_json(path_to_config, ModelTunerConfig)
# config.config_model_tuning.models = ["MLP_TargetTransformed"]
# config.config_model_tuning.hyperparameter_tuning_kwargs = {"n_trials": 50}
# config.config_model_tuning.hyperparameter_tuning_kwargs = {
#     "hyperparameter_set": {
#         "hidden_layer_sizes": [100, 100],
#         "activation": "relu",
#         "max_iter": 2000,
#     }
# }
#
# config.config_detector.detectors = ["KNN", "KDE", "GP", "OCSVM", "IF"]

# Configure the logger
result_folder = results_dir_extrapolation_experiment(config)
LocalLogger.directory = os.path.join(result_folder, "local_logger")
LocalLogger.active = True
WandbLogger.project = f"ED_{config.simulation_data_name}"
WandbLogger.directory = result_folder
WandbLogger.active = True

# Run scripts
ExperimentLogger.start_experiment(config=config)  # log config
s1_split_data.exe(config)
s2_tune_ml_regressor.exe(config)
s3_regressor_error_calculation.exe(config)
s4_true_validity_domain.exe(config)
s5_tune_detector.exe(config)
s6_detector_score_calculation.exe(config)
s8_1_coverage_convex_hull.exe(config)
s8_2_coverage_grid_occupancy.exe(config)
s8_3_coverage_tuned_ND.exe(config)
s8_4_coverage_true_validity.exe(config)
# s9_data_coverage.exe(config)
# s9_data_coverage_grid.exe(config)
#
# s7_2_plotting.exe_plot_2D_all(config)
# s7_2_plotting.exe_plot_2D_detector(config)

# from extrapolation_detection.use_cases import s9_data_coverage_debug
# s9_data_coverage_debug.exe(config)



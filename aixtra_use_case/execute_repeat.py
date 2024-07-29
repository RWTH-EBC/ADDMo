import os

from addmo.util.experiment_logger import ExperimentLogger
from addmo.util.experiment_logger import LocalLogger
from addmo.util.experiment_logger import WandbLogger
from addmo.util.definitions import create_or_clean_directory

from aixtra_use_case.config.ed_experiment_config import (
    ExtrapolationExperimentConfig,
)
from addmo.util.load_save import load_config_from_json
from aixtra_use_case import (
    s2_tune_ml_regressor,
    s4_true_validity_domain,
    s8_1_coverage_convex_hull,
    s3_regressor_error_calculation,
    s8_3_coverage_tuned_ND,
    s5_tune_detector,
    s1_split_data,
    s7_2_plotting,
    s8_4_coverage_true_validity,
    s6_detector_score_calculation,
    s8_2_coverage_grid_occupancy,
    s9_data_coverage_grid,
    s9_data_coverage,
    s8_0_generate_grid,
)

# load config from path
path_to_folder = r"D:\04_GitRepos\addmo-extra\aixtra_use_case\results\debug"
config_path = os.path.join(path_to_folder, "local_logger", "config.json")
config = load_config_from_json(config_path, ExtrapolationExperimentConfig)


# Configure the logger
create_or_clean_directory(config.experiment_folder)
LocalLogger.directory = os.path.join(config.experiment_folder, "local_logger")
LocalLogger.active = True
WandbLogger.project = f"TEST_{config.simulation_data_name}"
WandbLogger.directory = os.path.join(config.experiment_folder, "wandb_logger")
WandbLogger.active = False

config.experiment_name = "debug"
# config.grid_points_per_axis = 20
# config.config_explo_quant.explo_grid_points_per_axis = 20
# config.config_explo_quant.exploration_bounds = {
#     "TDryBul": [260.15, 286.15],
#     "HDirNor": [0, 1000],
#     "oveHeaPumY_u": [0, 1],
#     "reaTZon_y": [280.15, 295.15],
#     "delta_reaTZon_y": [-0.5, 0.5],
# }


# Run scripts
# ExperimentLogger.start_experiment(config=config)  # log config
# s1_split_data.exe(config)
# s2_tune_ml_regressor.exe(config)
# s3_regressor_error_calculation.exe(config)
# s4_true_validity_domain.exe(config)
# s5_tune_detector.exe(config)
# s6_detector_score_calculation.exe(config)

# s8_0_generate_grid.exe(config)
# s8_1_coverage_convex_hull.exe(config)
# s8_2_coverage_grid_occupancy.exe(config)
# s8_3_coverage_tuned_ND.exe(config)
# s8_4_coverage_true_validity.exe(config)
s9_data_coverage.exe(config)
s9_data_coverage_grid.exe(config)
# # #
# s7_2_plotting.exe_plot_2D_all(config)
# s7_2_plotting.exe_plot_2D_detector(config)

# from extrapolation_detection.aixtra_use_case import s9_data_coverage_debug
# s9_data_coverage_debug.exe(config)

ExperimentLogger.finish_experiment()

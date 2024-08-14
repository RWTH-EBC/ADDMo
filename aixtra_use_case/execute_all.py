import os

from keras.src.callbacks import EarlyStopping

from addmo.util.experiment_logger import ExperimentLogger
from addmo.util.experiment_logger import LocalLogger
from addmo.util.experiment_logger import WandbLogger
from addmo.util.definitions import create_or_clean_directory

from aixtra_use_case.config.ed_experiment_config import (
    ExtrapolationExperimentConfig,
)
from aixtra_use_case import (
    s1_split_data,
    s2_tune_ml_regressor,
    s3_regressor_error_calculation,
    s4_gradient,
    s4_true_validity_domain,
    s5_tune_detector,
    s6_detector_score_calculation,
    s7_2_plotting,
    s8_0_generate_grid,
    s8_1_coverage_convex_hull,
    s8_2_coverage_grid_occupancy,
    s8_3_coverage_tuned_ND,
    s8_4_coverage_true_validity,
    s8_5_coverage_gradient,
    s9_data_coverage,
    s9_data_coverage_grid,
    s9_carpet_plots,
)

# configure config
from aixtra_use_case.sweeps import config_blueprints, config_blueprints_systems
config = ExtrapolationExperimentConfig()
config = config_blueprints_systems.config_bes_steady(config)
config = config_blueprints.no_tuning_config(config)

# config.train_val_test_period = (23329, 26208, 32065, 35040) #September und Dezember


# Configure the logger
create_or_clean_directory(config.experiment_folder)
LocalLogger.directory = os.path.join(config.experiment_folder, "local_logger")
LocalLogger.active = True
WandbLogger.project = f"TEST_{config.simulation_data_name}"
WandbLogger.directory = os.path.join(config.experiment_folder, "wandb_logger")
WandbLogger.active = False


# Run scripts
ExperimentLogger.start_experiment(config=config)  # log config
s1_split_data.exe(config)
s2_tune_ml_regressor.exe(config)
s3_regressor_error_calculation.exe(config)
s4_true_validity_domain.exe(config)
s4_gradient.exe(config)
# s5_tune_detector.exe(config)
# s6_detector_score_calculation.exe(config)
s8_0_generate_grid.exe(config)
# # # s8_1_coverage_convex_hull.exe(config)
# # # s8_2_coverage_grid_occupancy.exe(config)
# # s8_3_coverage_tuned_ND.exe(config)
s8_4_coverage_true_validity.exe(config)
s8_5_coverage_gradient.exe(config)
# s9_data_coverage.exe(config)
s9_data_coverage_grid.exe(config)
s9_carpet_plots.exe(config)
# # #
# s7_2_plotting.exe_plot_2D_all(config)
# s7_2_plotting.exe_plot_2D_detector(config)

# from extrapolation_detection.aixtra_use_case import s9_data_coverage_debug
# s9_data_coverage_debug.exe(config)

ExperimentLogger.finish_experiment()

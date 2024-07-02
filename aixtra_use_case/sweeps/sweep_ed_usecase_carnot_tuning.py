import os

import wandb

from addmo.util.experiment_logger import ExperimentLogger
from addmo.util.experiment_logger import LocalLogger
from addmo.util.experiment_logger import WandbLogger
from addmo.util.definitions import create_or_clean_directory

from aixtra_use_case.sweeps import sweep_configs, config_blueprints

from aixtra_use_case.config.ed_experiment_config import (
    ExtrapolationExperimentConfig,
)
from aixtra_use_case import s2_tune_ml_regressor, s4_true_validity_domain, \
    s8_1_coverage_convex_hull, s3_regressor_error_calculation, s8_3_coverage_tuned_ND, \
    s5_tune_detector, s1_split_data, s7_2_plotting, s8_4_coverage_true_validity, \
    s6_detector_score_calculation, s8_2_coverage_grid_occupancy, s8_0_generate_grid


def define_config():
    # configure config
    config = ExtrapolationExperimentConfig()
    config.simulation_data_name = "Carnot_mid_noise_m0_std0.02"
    config.shuffle = False
    config.grid_points_per_axis = 100
    config.true_outlier_threshold = 0.2

    config.config_explo_quant.exploration_bounds = {
        "$T_{umg}$ in °C": (-10, 30),
        "$P_{el}$ in kW": (0, 5),
        "$\dot{Q}_{heiz}$ in kW": (0, 35)
    }

    config = config_function(config)

    return config


def run_all():
    # Define default values
    config = define_config()

    # Start wandb and overwrite default with sweep values where applicable
    run = wandb.init(config=config.dict())

    # update config with the experiment name of wandb run
    wandb.config.update(
        {"experiment_name": f"{project_name}_{run.name}"},
        allow_val_change=True,
    )

    # convert config dict back to pydantic object
    config = ExtrapolationExperimentConfig(**wandb.config)

    create_or_clean_directory(config.experiment_folder)
    LocalLogger.directory = os.path.join(config.experiment_folder, "local_logger")
    WandbLogger.directory = os.path.join(config.experiment_folder, "wandb_logger")

    # locally log the config as well
    LocalLogger.active = True
    WandbLogger.active = False
    ExperimentLogger.start_experiment(config=config)  # log config

    # activate wandb logging again
    WandbLogger.active = True

    # Run scripts
    s1_split_data.exe(config)
    s2_tune_ml_regressor.exe(config)
    s3_regressor_error_calculation.exe(config)
    s4_true_validity_domain.exe(config)
    s5_tune_detector.exe(config)
    s6_detector_score_calculation.exe(config)
    s8_0_generate_grid.exe(config)
    s8_1_coverage_convex_hull.exe(config)
    s8_2_coverage_grid_occupancy.exe(config)
    s8_3_coverage_tuned_ND.exe(config)
    s8_4_coverage_true_validity.exe(config)

    s7_2_plotting.exe_plot_2D_all(config)
    s7_2_plotting.exe_plot_2D_detector(config)

    ExperimentLogger.finish_experiment()


####################################################################################################
config_function = config_blueprints.tuning_config() # Todo: Set

config_temp = define_config()

project_name = f"3_{config_temp.simulation_data_name}" #Todo: Set
# project_name = "Test"

# sweep
sweep_configuration = sweep_configs.sweep_several_tunings() #Todo: Set

sweep_id = wandb.sweep(sweep_configuration, project=project_name)
wandb.agent(sweep_id, function=run_all, project=project_name)


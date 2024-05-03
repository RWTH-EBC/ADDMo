import os

import wandb

from core.util.experiment_logger import ExperimentLogger
from core.util.experiment_logger import LocalLogger
from core.util.experiment_logger import WandbLogger
from core.util.definitions import root_dir
from core.util.load_save import load_config_from_json
from core.util.definitions import results_dir_extrapolation_experiment
from core.s3_model_tuning.config.model_tuning_config import ModelTunerConfig

from sweeps import sweep_configs
from sweeps import config_blueprints

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

    config = config_blueprints.linear_regression_config(config)

    return config


def run_all():
    # Define default values
    config = define_config()

    # Start wandb and overwrite default with sweep values where applicable
    run = wandb.init(config=config.dict())

    # update config with the experiment name of wandb run
    wandb.config.update(
        {"experiment_name": f"1_{config.simulation_data_name}_{run.name}"},
        allow_val_change=True,
    )

    # convert config dict back to pydantic object
    config = ExtrapolationExperimentConfig(**wandb.config)

    result_folder = results_dir_extrapolation_experiment(config.experiment_name)
    LocalLogger.directory = os.path.join(result_folder, "local_logger")
    WandbLogger.directory = result_folder

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
    s8_1_coverage_convex_hull.exe(config)
    s8_2_coverage_grid_occupancy.exe(config)
    s8_3_coverage_tuned_ND.exe(config)
    s8_4_coverage_true_validity.exe(config)

    s7_2_plotting.exe_plot_2D_all(config)
    s7_2_plotting.exe_plot_2D_detector(config)

    ExperimentLogger.finish_experiment()


config_temp = define_config()


project_name = f"2_{config_temp.simulation_data_name}"
# project_name = "Test"

# sweep
sweep_configuration = sweep_configs.sweep_repetitions_only()
sweep_id = wandb.sweep(sweep_configuration, project=project_name)
wandb.agent(sweep_id, function=run_all, project=project_name)


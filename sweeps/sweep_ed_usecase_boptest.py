import os

import wandb

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


def define_config():
    # configure config
    config = ExtrapolationExperimentConfig()
    config.simulation_data_name = "Boptest_TAir_mid_ODE_noise_m0_std0.01"
    config.experiment_name = "Empty"
    config.name_of_target = "delta_reaTZon_y"
    config.train_val_test_period = (0, 1488)
    config.shuffle = False
    config.grid_points_per_axis = 10
    config.system_simulation = "BopTest_TAir_ODE"  # "carnot
    config.true_outlier_threshold = 0.1
    #
    config.config_explo_quant.explo_grid_points_per_axis = 10

    path_to_config = os.path.join(
        root_dir(),
        "core",
        "s3_model_tuning",
        "config",
        "model_tuner_config_no_tuning.json",
    )
    config.config_model_tuning = load_config_from_json(path_to_config, ModelTunerConfig)
    config.config_model_tuning.models = ["MLP_TargetTransformed"]
    config.config_model_tuning.hyperparameter_tuning_kwargs = {
        "hyperparameter_set": {
            "hidden_layer_sizes": [5],
            "activation": "relu",
            "max_iter": 2000,
        }
    }

    config.config_detector.detectors = ["KNN", "GP", "OCSVM"]
    return config


def run_all():
    # Define default values
    config = define_config()

    # Start wandb and overwrite default with sweep values where applicable
    run = wandb.init(config=config.dict())

    # update config with the experiment name of wandb run
    wandb.config.update({"experiment_name": f"{config.simulation_data_name}_1_{run.name}"}, allow_val_change=True)

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

    # s7_2_plotting.exe_plot_2D_all(config)
    # s7_2_plotting.exe_plot_2D_detector(config)

    ExperimentLogger.finish_experiment()


hidden_layer_sizes = []

# Single layer possibilities
for neurons in [5, 10, 100, 1000]:
    hidden_layer_sizes.append([neurons])

# Two layer possibilities
for neurons1 in [5, 10, 100, 1000]:
    for neurons2 in [5, 10, 100, 1000]:
        hidden_layer_sizes.append([neurons1, neurons2])

sweep_configuration = {
    "name": "trial_sweep_ed_usecase",
    "method": "grid",
    "metric": {"name": "coverage_true_validity", "goal": "maximize"},
    "parameters": {
        "repetition": {"values": [1, 2, 3, 4, 5, 6, 7, 8]},
        "config_model_tuning": {
            "parameters": {
                "hyperparameter_tuning_kwargs": {
                    "parameters": {
                        "hyperparameter_set": {
                            "parameters": {
                                "hidden_layer_sizes": {"values": hidden_layer_sizes}
                            }
                        }
                    }
                }
            }
        },
    },
}

config_temp = define_config()

# project_name = "Test"
project_name = config_temp.simulation_data_name

sweep_id = wandb.sweep(sweep_configuration, project=project_name)

wandb.agent(sweep_id, function=run_all, project=project_name)
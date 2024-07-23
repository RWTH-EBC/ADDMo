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
    s5_tune_detector, s1_split_data, s8_4_coverage_true_validity, s6_detector_score_calculation, \
    s8_2_coverage_grid_occupancy, s8_0_generate_grid, s9_data_coverage_grid, s9_data_coverage


def define_config(config_function):
    # configure config
    config = ExtrapolationExperimentConfig()
    config.simulation_data_name = "ODEel_steady"
    config.experiment_name = "Empty"
    config.name_of_target = "delta_reaTZon_y"
    config.train_val_test_period = (29185, 35040) #November und Dezember
    config.shuffle = False
    config.grid_points_per_axis = 10
    config.system_simulation = "BopTest_TAir_ODEel"
    config.true_outlier_threshold = 0.1

    config.config_explo_quant.exploration_bounds = {
        "TDryBul": (263.15, 303.15),
        "HDirNor": (0, 1000),
        "oveHeaPumY_u": (0, 1),
        "reaTZon_y": (290.15, 300.15),
        "delta_reaTZon_y": (-0.5, 0.5),
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

    if config.config_model_tuning.models == ["ScikitLinearRegNoScaler"]:
        wandb.run.name = f"LinReg_{run.name}"

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
    s9_data_coverage.exe(config)
    s9_data_coverage_grid.exe(config)

    ExperimentLogger.finish_experiment()


####################################################################################################
def main():
    config_function = config_blueprints.linear_regression_config # Todo: Set without brackets

    config_temp = define_config(config_function)

    project_name = f"5_{config_temp.simulation_data_name}" #Todo: Set
    # project_name = "Test"

    # sweep
    sweep_configuration = sweep_configs.sweep_repetitions_only() #Todo: Set

    sweep_id = wandb.sweep(sweep_configuration, project=project_name)
    wandb.agent(sweep_id, function=run_all, project=project_name)

if __name__ == '__main__':
    main()

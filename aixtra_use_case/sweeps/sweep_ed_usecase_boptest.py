import os
import wandb
from addmo.util.experiment_logger import ExperimentLogger, LocalLogger, WandbLogger
from addmo.util.definitions import create_or_clean_directory, root_dir
from aixtra_use_case.sweeps import (
    sweep_configs,
    config_blueprints,
    config_blueprints_systems,
)
from aixtra_use_case.config.ed_experiment_config import ExtrapolationExperimentConfig
from aixtra_use_case import (
    s1_split_data,
    s2_tune_ml_regressor,
    s3_regressor_error_calculation,
    s4_true_validity_domain,
    s5_tune_detector,
    s6_detector_score_calculation,
    s8_0_generate_grid,
    s8_1_coverage_convex_hull,
    s8_2_coverage_grid_occupancy,
    s8_3_coverage_tuned_ND,
    s8_4_coverage_true_validity,
)


def run_all():
    # create config
    config = create_config()

    # merge config with sweep values
    wandb.init(config=config.dict())

    # convert config dict back to pydantic object
    config = ExtrapolationExperimentConfig(**wandb.config)

    # Set up experiment name and folders
    if config.config_model_tuning.models == ["ScikitLinearRegNoScaler"]:
        wandb.run.name = f"LinReg_{wandb.run.name}"

    # update config
    run_name = f"{config.experiment_name}_{wandb.run.name}"
    config.experiment_name = run_name
    wandb.config.update(
        {"experiment_name": run_name},
        allow_val_change=True,
    )

    create_or_clean_directory(config.experiment_folder)
    LocalLogger.directory = os.path.join(config.experiment_folder, "local_logger")
    WandbLogger.directory = os.path.join(config.experiment_folder, "wandb_logger")

    # Set up logging
    LocalLogger.active = True
    WandbLogger.active = False
    ExperimentLogger.start_experiment(config=config)
    WandbLogger.active = True

    # Run scripts
    s1_split_data.exe(config)
    s2_tune_ml_regressor.exe(config)
    s3_regressor_error_calculation.exe(config)
    s4_true_validity_domain.exe(config)
    s5_tune_detector.exe(config)
    s6_detector_score_calculation.exe(config)
    s8_0_generate_grid.exe(config)
    # s8_1_coverage_convex_hull.exe(config)
    # s8_2_coverage_grid_occupancy.exe(config)
    s8_3_coverage_tuned_ND.exe(config)
    s8_4_coverage_true_validity.exe(config)

    ExperimentLogger.finish_experiment()


def create_config():  # Todo set
    config = ExtrapolationExperimentConfig()
    config = config_blueprints_systems.config_ODEel_steady(config)
    config = config_blueprints.linear_regression_config(config)
    config.experiment_name = f"7_{config.simulation_data_name}_NovDez"
    return config


def create_sweep():
    config = create_config()

    sweep_configuration = sweep_configs.sweep_repetitions_only()  # Todo set

    entity = wandb.api.default_entity
    project_name = config.experiment_name
    sweep_id = wandb.sweep(sweep_configuration, project=project_name)  # Initialize the sweep

    full_sweep_path = f"{entity}/{project_name}/{sweep_id}"

    print(f"Sweep created: {full_sweep_path}")
    print(f"To run an agent in your pycharm console, use the following Python code:")
    print(f"from aixtra_use_case.sweeps.sweep_ed_usecase_boptest import run_all")
    print(f"import wandb")
    print(f"wandb.agent('{full_sweep_path}', function=run_all)")

    print("\nTo run an agent from the Anaconda command prompt, follow these steps:")
    print("   conda activate ADDMo-Extra")
    print("   conda activate py311_addmoextra")
    print(f"   cd /d {root_dir()}")
    print(f"   python -m aixtra_use_case.sweeps.sweep_ed_usecase_boptest {full_sweep_path}")

    return sweep_id


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # If a sweep ID is provided, run as an agent
        wandb.agent(sys.argv[1], function=run_all)
    else:
        # If no sweep ID is provided, create a new sweep
        sweep_id = create_sweep()
        wandb.agent(sweep_id, function=run_all)

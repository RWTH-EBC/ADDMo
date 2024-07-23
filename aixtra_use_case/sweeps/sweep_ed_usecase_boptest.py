import os
import wandb
from addmo.util.experiment_logger import ExperimentLogger, LocalLogger, WandbLogger
from addmo.util.definitions import create_or_clean_directory, root_dir
from aixtra_use_case.sweeps import sweep_configs, config_blueprints, config_blueprints_systems
from aixtra_use_case.config.ed_experiment_config import ExtrapolationExperimentConfig
from aixtra_use_case import (
    s1_split_data, s2_tune_ml_regressor, s3_regressor_error_calculation,
    s4_true_validity_domain, s5_tune_detector, s6_detector_score_calculation,
    s8_0_generate_grid, s8_1_coverage_convex_hull, s8_2_coverage_grid_occupancy,
    s8_3_coverage_tuned_ND, s8_4_coverage_true_validity
)



def run_all():
    # Start wandb and get the config
    run = wandb.init()
    config = ExtrapolationExperimentConfig(**wandb.config)

    # Set up experiment name and folders
    if config.config_model_tuning.models == ["ScikitLinearRegNoScaler"]:
        wandb.run.name = f"LinReg_{wandb.run.name}"

    # update config
    config.experiment_name = f"{config.experiment_name}_{wandb.run.name}"

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


def create_sweep():
    # Set up the base configuration
    config = ExtrapolationExperimentConfig()
    config = config_blueprints_systems.config_ODEel_steady(config)
    config = config_blueprints.linear_regression_config(config)
    config.config_detector.detectors = ["KNN_untuned"]

    # Set up the project name
    project_name = f"TEST_{config.simulation_data_name}"

    config.experiment_name = project_name

    # Set up the sweep configuration
    sweep_configuration = sweep_configs.sweep_repetitions_only()


    # Add the config parameters to the sweep configuration
    for key, value in config.dict().items():
        if key not in sweep_configuration['parameters']:
            sweep_configuration['parameters'][key] = {'value': value}

    # Initialize the sweep
    sweep_id = wandb.sweep(sweep_configuration, project=project_name)


    print(f"Sweep created with ID: {sweep_id}")
    print(f"Project name: {project_name}")
    print(f"To run an agent in your pycharm console, use the following Python code:")
    print(f"from aixtra_use_case.sweeps.sweep_ed_usecase_boptest import run_all")
    print(f"import wandb")
    print(f"wandb.agent('{sweep_id}', function=run_all, project='{project_name}')")

    print("\nTo run an agent from the Anaconda command prompt, follow these steps:")
    print("   conda activate ADDMo-Extra")
    print(f"   cd /d {root_dir()}")
    print(f"   python -m aixtra_use_case.sweeps.sweep_ed_usecase_boptest {sweep_id} {project_name}")

    # info to run through multiprocessing python script
    # Get the full sweep path
    entity = wandb.api.default_entity
    full_sweep_path = f"{entity}/{project_name}/{sweep_id}"
    print("\nTo run an agent from a multiprocessing Python script, copy the sweep ID:")
    print(f"   {full_sweep_path}")

    return sweep_id


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # If a sweep ID is provided, run as an agent
        wandb.agent(sys.argv[1], function=run_all, project=sys.argv[2])
    else:
        # If no sweep ID is provided, create a new sweep
        sweep_id = create_sweep()
        # wandb.agent(sweep_id, function=run_all)
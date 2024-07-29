import wandb
from typing import Dict, Any, List, Tuple


def update_run(api: wandb.Api, run_path: str, summary_dict: Dict[str, Any],
               config_dict: Dict[str, Any]) -> None:
    """Update the summary and config of a single run with multiple key-value pairs."""
    try:
        run = api.run(run_path)
        if summary_dict:
            run.summary.update(summary_dict)
            run.summary.update()  # Explicitly update to ensure changes are saved
            print(f"Updated run {run_path} summary with: {summary_dict}")
        if config_dict:
            for key, value in config_dict.items():
                run.config[key] = value
            run.update()  # Save the changes
            print(f"Updated run {run_path} config with: {config_dict}")
    except Exception as e:
        print(f"Error updating run {run_path}: {str(e)}")


def batch_update_runs(run_paths: List[str], summary_dict: Dict[str, Any],
                      config_dict: Dict[str, Any]) -> None:
    """Update multiple runs with the same summary and config updates."""
    api = wandb.Api()
    for run_path in run_paths:
        update_run(api, run_path, summary_dict, config_dict)


def get_user_confirmation(run_paths: List[str], summary_dict: Dict[str, Any],
                          config_dict: Dict[str, Any]) -> bool:
    """Display warning and get user confirmation."""
    print("\nWARNING: You are about to update the following runs:")
    for run_path in run_paths:
        print(f"- {run_path}")

    if summary_dict:
        print("\nThe following summary keys will be updated:")
        for key, value in summary_dict.items():
            print(f"- {key}: {value}")

    if config_dict:
        print("\nThe following config keys will be updated:")
        for key, value in config_dict.items():
            print(f"- {key}: {value}")

    confirmation = input("\nDo you want to proceed with these updates? (y/n): ").strip().lower()
    return confirmation == 'y'


if __name__ == '__main__':
    RUN_PATHS = [
        "team-martinraetz/6_ODEel_steady_NovDez___MPC_Typ2D/r5cez6iq"
    ]

    SUMMARY_UPDATE_DICT = {
                "hidden_layer_sizes": "[]",
                "model_complexity": 0,
                "best_model_name": "whitebox",
                "mean_errors_grid": 0,
                "mean_errors_remaining": 0,
                "mean_errors_test": 0,
                "mean_errors_train": 0,
                "mean_errors_val": 0,
                "true_valid_fraction_grid": 1,
                "true_valid_fraction_remaining": 1,
                "true_valid_fraction_test": 1,
                "true_valid_fraction_train": 1,
                "true_valid_fraction_val": 1,
                "coverage_true_validity": 100,
            }

    CONFIG_UPDATE_DICT = {
        # "config_model_tuning.validation_score_splitting": "None"
    }

    if get_user_confirmation(RUN_PATHS, SUMMARY_UPDATE_DICT, CONFIG_UPDATE_DICT):
        batch_update_runs(RUN_PATHS, SUMMARY_UPDATE_DICT, CONFIG_UPDATE_DICT)
        print("Finished updating all specified runs.")
    else:
        print("Update cancelled.")
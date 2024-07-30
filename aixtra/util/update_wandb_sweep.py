import wandb
from typing import Dict, Any, Iterator, List


def yield_runs_per_sweep(user_name: str, project_name: str, sweep_id: str) -> Iterator[
    wandb.apis.public.Run]:
    """Fetch the runs from a sweep."""
    api = wandb.Api()
    sweep_path = f"{user_name}/{project_name}/{sweep_id}"
    print(f"Fetching sweep: {sweep_path}")
    sweep = api.sweep(sweep_path)
    yield from sweep.runs


def update_run(run: wandb.apis.public.Run, summary_dict: Dict[str, Any],
               config_dict: Dict[str, Any]) -> None:
    """Update the summary and config of a single run."""
    try:
        if summary_dict:
            run.summary.update(summary_dict)
            run.summary.update()  # Explicitly update to ensure changes are saved
            print(f"Updated run {run.id} summary with: {summary_dict}")

        if config_dict:
            for key, value in config_dict.items():
                keys = key.split('.')
                if len(keys) == 1:
                    run.config[keys[0]] = value
                elif len(keys) == 2:
                    run.config[keys[0]][keys[1]] = value
                elif len(keys) == 3:
                    run.config[keys[0]][keys[1]][keys[2]] = value
                elif len(keys) == 4:
                    run.config[keys[0]][keys[1]][keys[2]][keys[3]] = value
                else:
                    print(f"Warning: Key '{key}' has more than 4 levels of nesting. Skipping.")
            run.update()  # Save the changes
            print(f"Updated run {run.id} config with: {config_dict}")

    except Exception as e:
        print(f"Error updating run {run.id}: {str(e)}")


def get_user_confirmation(runs: List[wandb.apis.public.Run], summary_dict: Dict[str, Any],
                          config_dict: Dict[str, Any]) -> bool:
    """Display warning and get user confirmation."""
    print("\nWARNING: You are about to update the following runs:")
    for run in runs:
        print(f"- {run.name}")

    print("\nThe following keys will be updated:")
    print("Summary keys:")
    for key in summary_dict.keys():
        print(f"- {key}")
    print("Config keys:")
    for key in config_dict.keys():
        print(f"- {key}")

    confirmation = input("\nDo you want to proceed with these updates? (y/n): ").strip().lower()
    return confirmation == 'y'


def batch_update_sweep_runs(user_name: str, project_name: str, sweep_id: str,
                            summary_dict: Dict[str, Any], config_dict: Dict[str, Any]) -> None:
    """Update all runs in a sweep with the same summary and config updates after confirmation."""
    runs = list(yield_runs_per_sweep(user_name, project_name, sweep_id))

    if get_user_confirmation(runs, summary_dict, config_dict):
        for run in runs:
            update_run(run, summary_dict, config_dict)
        print("Finished updating all runs in the sweep.")
    else:
        print("Update cancelled.")


if __name__ == '__main__':
    USER_NAME = "team-martinraetz"
    PROJECT_NAME = "7_Carnot_mid_noise_m0_std0.02"
    SWEEP_ID = "xyjgjlew"



    SUMMARY_UPDATE_DICT = {}
    #     "hidden_layer_sizes": "[]",
    #     "model_complexity": 1,
    # }

    CONFIG_UPDATE_DICT = {
        "config_model_tuning.validation_score_splitting": "no"
    }

    batch_update_sweep_runs(USER_NAME, PROJECT_NAME, SWEEP_ID, SUMMARY_UPDATE_DICT,
                            CONFIG_UPDATE_DICT)
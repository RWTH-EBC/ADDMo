import wandb
from typing import Dict, Any, List


def update_run_summary(api: wandb.Api, run_path: str, update_dict: Dict[str, Any]) -> None:
    """Update the summary of a single run with multiple key-value pairs."""
    try:
        run = api.run(run_path)
        run.summary.update(update_dict)
        print(f"Updated run {run_path} with: {update_dict}")
    except Exception as e:
        print(f"Error updating run {run_path}: {str(e)}")


def batch_update_runs(run_paths: List[str], update_dict: Dict[str, Any]) -> None:
    """Update multiple runs with the same summary updates."""
    api = wandb.Api()
    for run_path in run_paths:
        update_run_summary(api, run_path, update_dict)


if __name__ == '__main__':
    RUN_PATHS = [
        "team-martinraetz/5_ODEel_steady/l5pqwg2e"
    ]

    UPDATE_DICT = {
        "hidden_layer_sizes": "[]",
        "model_complexity": 0
    }

    batch_update_runs(RUN_PATHS, UPDATE_DICT)

    print("Finished updating all specified runs.")
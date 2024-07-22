import wandb
from typing import Dict, Any, Iterator


def yield_runs_per_sweep(user_name: str, project_name: str, sweep_id: str) -> Iterator[
    wandb.apis.public.Run]:
    """Fetch the runs from a sweep."""
    api = wandb.Api()
    sweep_path = f"{user_name}/{project_name}/{sweep_id}"
    print(f"Fetching sweep: {sweep_path}")
    sweep = api.sweep(sweep_path)
    yield from sweep.runs


def update_run_summary(run: wandb.apis.public.Run, update_dict: Dict[str, Any]) -> None:
    """Update the summary of a single run with multiple key-value pairs."""
    try:
        run.summary.update(update_dict)
        print(f"Updated run {run.id} with: {update_dict}")
    except Exception as e:
        print(f"Error updating run {run.id}: {str(e)}")


def batch_update_sweep_runs(user_name: str, project_name: str, sweep_id: str,
                            update_dict: Dict[str, Any]) -> None:
    """Update all runs in a sweep with the same summary updates."""
    for run in yield_runs_per_sweep(user_name, project_name, sweep_id):
        update_run_summary(run, update_dict)


if __name__ == '__main__':
    USER_NAME = "team-martinraetz"
    PROJECT_NAME = "5_ODEel_steady"
    SWEEP_ID = "fns7vs0n"

    UPDATE_DICT = {
        "hidden_layer_sizes": "[]",
        "model_complexity": 0
    }

    batch_update_sweep_runs(USER_NAME, PROJECT_NAME, SWEEP_ID, UPDATE_DICT)

    print("Finished updating all runs in the sweep.")
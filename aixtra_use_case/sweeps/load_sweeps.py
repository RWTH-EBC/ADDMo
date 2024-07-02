import wandb
import pandas as pd
from addmo.s3_model_tuning.models.model_factory import ModelFactory

#Todo: clean up: maybe move some general functions to utils? Depending on what we have in DDMPC Repo.

def yield_runs_per_sweep(project_name, sweep_id):
    # Initialize the API
    api = wandb.Api()

    # Fetch the sweep
    sweep_path = f"rishika-eon/{project_name}/{sweep_id}"
    print(f"Fetching sweep: {sweep_path}")
    sweep = api.sweep(sweep_path)

    for run in sweep.runs:
        yield run


def load_artifact(run, artifact_name, artifact_type):
    api = wandb.Api()

    # Fetch the artifact
    artifact = api.artifact(f'{run.entity}/{run.project}/{artifact_name}:{artifact_type}')
    artifact_dir = artifact.download()
    return artifact_dir


def load_data_from_drive(run):
    pass


def load_logs(run):
    # Fetch the logs (history) of the run
    history = run.history()
    return history


def get_best_run(runs, metric_name):
    best_run = max(runs, key=lambda run: run.summary.get(metric_name))
    best_score = best_run.summary.get(metric_name)
    return best_run, best_score


def main():
    # Login to wandb
    wandb.login()
    project_name = '1_Boptest_TAir_mid_ODE_noise_m0_std0.01'
    sweep_id = "shd6lg5t"
    metric_name = "optuna_study_best_validation_score"

    summary_list, config_list, name_list = [], [], []

    for run in yield_runs_per_sweep(project_name, sweep_id):

        name = run.name
        print(f"Processing run: {name}")
        summary_logs = run.summary._json_dict
        config = run.config
        summary_list.append(summary_logs)
        config_list.append({k: v for k, v in config.items() if not k.startswith("_")})
        name_list.append(run.name)
        # Load artifacts and log
        # alias = 'latest'
        # artifact_dir = load_artifact(run, 'optuna_study', alias)
        # regressor = load_artifact(run, 'regressor', alias)
        # history = load_logs(run)

    # Save run metadata
    runs_df = pd.DataFrame({"summary": summary_list, "config": config_list, "name": name_list})
    runs_df.to_csv("Sweep_summary.csv")

    # Downloading the best model:

    runs = list(yield_runs_per_sweep(project_name, sweep_id))
    best_run, best_score = get_best_run(runs, metric_name)
    print(f"Best run: {best_run.name} with {best_score} validation score")
    # Not using load_artifact here since can't fetch alias dynamically
    artifacts = best_run.logged_artifacts()
    best_model = [artifact for artifact in artifacts if artifact.type == 'keras'][0]
    dir = best_model.download()
    regressor = ModelFactory.load_model(f"{dir}/regressor1.keras")
    regressor.regressor.model.summary()

if __name__ == "__main__":
    main()

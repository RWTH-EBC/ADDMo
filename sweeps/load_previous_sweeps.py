import wandb


def yield_runs_per_sweep(project_name, sweep_id):
    # Initialize the API
    api = wandb.Api()

    # Fetch the sweep
    sweep_path = f"rishika-eon/{project_name}/{sweep_id}"
    print(f"Fetching sweep: {sweep_path}")
    sweep = api.sweep(sweep_path)

    for run in sweep.runs:
        print(f"Yielding run: {run.name}")
        yield run

        # info to retrieve data from run
        # summary_logs = run.summary._json_dict
        # run_name = run.name
        # run_config = run.config
        # model = run.loadmodel()


def load_artifact(run, artifact_name, artifact_type):
    api = wandb.Api()

    # Fetch the artifact
    artifact = api.artifact(f'{run.entity}/{run.project}/{artifact_name}:{artifact_type}')
    # artifact = run.use_artifact(f'{artifact_name}:{artifact_type}')
    artifact_dir = artifact.download()
    return artifact_dir


def load_data_from_drive(run):
    pass


def load_logs(run):
    # Fetch the logs (history) of the run
    history = run.history()
    return history


def main():
    # Login to wandb
    wandb.login()

    project_name = '1_Boptest_TAir_mid_ODE_noise_m0_std0.01'
    sweep_id =  ["gsz2f0ez"] #, "pdj7zco"]
    for sweep_id in sweep_id:
        print(f"Processing sweep: {sweep_id}")
    # Iterate through the runs in the sweep and collect logs and artifacts
        for run in yield_runs_per_sweep(project_name, sweep_id):
        # Fetch the logs (history) of the run
            summary_logs = run.summary._json_dict
            name = run.name
            config = run.config
            print(f"Processing run: {name}")

    # Load artifacts if any
            artifact_name = 'optuna_study'
            artifact_type = 'latest'
            artifact_dir = load_artifact(run, artifact_name, artifact_type)
            print(f"Artifact downloaded to: {artifact_dir}")

            print(f"The config is : {config}")
            print(f"summary is : {summary_logs}")
    # Load logs
            history = load_logs(run)
            print(f"History logs for run {name}: {history}")


if __name__ == "__main__":
    main()

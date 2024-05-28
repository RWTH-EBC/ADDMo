import wandb


def yield_runs_per_sweep(project_name, sweep_id):
    # Initialize the API
    api = wandb.Api()

    # Fetch the sweep
    sweep_path = f"team-martinraetz/{project_name}/{sweep_id}"
    print(f"Fetching sweep: {sweep_path}")
    sweep = api.sweep(sweep_path)

    for run in sweep.runs:
        yield run


def main():
    # Login to wandb
    wandb.login()

    project_name = '1_Boptest_TAir_mid_ODE_noise_m0_std0.01'
    sweep_id = "aspiv35n"

    # Iterate through the runs in the sweep and collect logs and artifacts
    for run in yield_runs_per_sweep(project_name, sweep_id):
        # Fetch the logs (history) of the run
        summary_logs = run.summary._json_dict
        name = run.name

if __name__ == "__main__":
    main()

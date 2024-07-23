import time
import wandb


def main():
    # Initialize a wandb run
    run = wandb.init()
    config = run.config

    # Access the wait_time and run_name from the configuration
    wait_time = config.get("wait_time")
    run_name = config.get("run_name")

    # Wait for the specified amount of time
    time.sleep(wait_time)

    # Print the run name and log it to wandb
    print(f"{run_name} executed")
    wandb.log({"run_name": run_name, "status": "executed"})


if __name__ == "__main__":
    main()
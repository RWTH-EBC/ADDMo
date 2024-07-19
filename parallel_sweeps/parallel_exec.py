# import subprocess
# import os
#
# def run_agent(sweep_id):
#     try:
#         result = subprocess.run(['wandb', 'agent', sweep_id])
#         print(result.stdout)
#         if result.returncode != 0:
#             print(result.stderr)
#             raise RuntimeError(f"W&B agent failed with exit code {result.returncode}")
#     except Exception as e:
#         print(f"An error occurred while running the W&B agent: {e}")
#
# def main():
#     os.environ['WANDB_NETWORK_TIMEOUT'] = '300'
#     sweep_id = "rishika-eon/parallel_sweeps/s8he29l5"
#     num_agents = 2
#
#     processes = []
#     for _ in range(num_agents):
#         p = subprocess.Popen(['wandb', 'agent', sweep_id])
#         processes.append(p)
#
#     # Wait for all agents to complete
#     for p in processes:
#         p.wait()
#
# if __name__ == "__main__":
#     main()

import wandb
import multiprocessing
import time


# Function to run a W&B agent
def run_agent(sweep_id):
    wandb.agent(sweep_id)


if __name__ == '__main__':
    # Define your sweep ID
    sweep_id = 'rishika-eon/parallel_sweeps/mbquv8y8'

    # Number of parallel runs you want
    num_parallel_runs = 2

    # Create a pool of workers
    pool = multiprocessing.Pool(num_parallel_runs)

    # Run the agents
    for _ in range(num_parallel_runs):
        pool.apply_async(run_agent, args=(sweep_id,))

    # Close the pool and wait for the work to finish
    pool.close()
    pool.join()

    print("All agents have completed their runs.")

import wandb
import multiprocessing
import time


# Function to run a W&B agent
def run_agent(sweep_id):
    wandb.agent(sweep_id)


if __name__ == '__main__':
    # Define your sweep ID
    sweep_id = 'team-martinraetz/TEST_ODEel_steady/hypykwqd'

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

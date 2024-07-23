import wandb

def get_initial_sweep_config():
    sweep_config = {
        'method': 'grid',
        'parameters': {
            'wait_time': {
                'values': [10]
            },
            'run_name': {
                'values': ["run_1", "run_2", "run_3", "run_4", "run_5"]
            }
        },
        'program': r'C:\Users\mre-rpa\PycharmProjects\pythonProject2\addmo-automated-ml-regression\parallel_sweeps\exec.py'
    }
    return sweep_config

def create_sweep():
    config = get_initial_sweep_config()
    sweep_id = wandb.sweep(config, project="parallel_sweeps",  entity="rishika-eon")
    return sweep_id

if __name__ == "__main__":
    sweep_id = create_sweep()
    print(f"Sweep created with ID: {sweep_id}")
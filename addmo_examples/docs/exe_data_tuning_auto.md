
```python
import os
import json
import pandas as pd
from addmo.util.plotting_utils import save_pdf
from addmo.util.definitions import results_dir_data_tuning
from addmo.util.load_save_utils import root_dir
from addmo.util.experiment_logger import ExperimentLogger
from addmo.util.experiment_logger import LocalLogger
from addmo.util.experiment_logger import WandbLogger
from addmo.s1_data_tuning_auto.config.data_tuning_auto_config import DataTuningAutoSetup
from addmo.s1_data_tuning_auto.data_tuner_auto import DataTunerAuto
from addmo.s5_insights.model_plots.time_series import plot_timeseries_combined
from addmo.util.load_save import load_config_from_json

def _exe_data_tuning_auto(config, user_input='y'):
    """
    Execute the system_data tuning process automatically.
    """

    # Configure the logger
    LocalLogger.active = True
    if LocalLogger.active:
        LocalLogger.directory = results_dir_data_tuning(config, user_input)
    WandbLogger.project = "addmo-test_data_auto_tuning"
    WandbLogger.active = False
    if WandbLogger.active:
        WandbLogger.directory = results_dir_data_tuning(config,user_input)

    # Initialize logging
    ExperimentLogger.start_experiment(config=config)

    # Create the system_data tuner
    tuner = DataTunerAuto(config=config)

    # Tune the system_data
    tuned_x = tuner.tune_auto()
    y = tuner.y

    tuned_xy = pd.concat([y, tuned_x], axis=1, join="inner").bfill()

    # Log the tuned system_data
    file_name = 'tuned_xy_auto'
    ExperimentLogger.log_artifact(tuned_xy, file_name, art_type='system_data')

    # Return file paths for plotting data
    saved_data_path = os.path.join(LocalLogger.directory, file_name + '.csv')
    data = pd.read_csv(saved_data_path, delimiter=",", index_col=0, encoding="latin1", header=0)
    config_path = os.path.join(LocalLogger.directory, "config.json")
    with open(config_path, 'r') as f:
        plot_config = json.load(f)

    # Plot tuned data
    figures = plot_timeseries_combined(plot_config, data)
    for fig in figures:
        fig.show()
    os.makedirs(LocalLogger.directory, exist_ok=True)
    for idx, fig in enumerate(figures):
        suffix = "_2weeks" if idx == 1 else ""
        plot_path = os.path.join(LocalLogger.directory, f"{file_name}{suffix}")
        save_pdf(fig, plot_path)
    print("Finished")
```

Please define the missing TODOs in the section below according to the docstrings.

```python
"""
Execute the system_data tuning process with user defined config.
 Parameters:
    user_input : str, optional
        If 'y', the contents of the target results directory will be overwritten.
        If 'd', the directory contents will be deleted. Default is 'y'.
"""
user_input = 'y'
```

Path to the config file

```python
path_to_config = os.path.join(root_dir(), 'addmo', 's1_data_tuning_auto', 'config',
                              'data_tuning_auto_config.json')
```

Create the config object

```python
config = load_config_from_json(path_to_config, DataTuningAutoSetup)
```

Run data tuning execution

```python
_exe_data_tuning_auto(config, user_input=user_input)

default_config_exe_data_tuning_auto(user_input='y'):
"""Execute the system_data tuning process with default config."""
```

Initialize a default config (without loading JSON)

```python
config = DataTuningAutoSetup()
```

Run data tuning execution

```python
_exe_data_tuning_auto(config, user_input=user_input)
```

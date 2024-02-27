import wandb
import pandas as pd

from core.util.definitions import root_dir, results_dir_model_tuning
from core.util.experiment_logger import LocalLogger
from core.util.experiment_logger import WandbLogger
from core.util.experiment_logger import ExperimentLogger

from core.s3_model_tuning.config.model_tuning_config import ModelTuningSetup
from core.s3_model_tuning.model_tuner import ModelTuner
from core.util.load_save import load_data
from core.util.data_handling import split_target_features


# Load the config from the yaml file
def function_sweep(config=None):
    exe_data_tuning(config.data_tuning)
    exe_model_tuning(config.model_tuning)


wandb.agent("2vpa873o", function=function_sweep, project="addmo-test_model_tuning", count=1)
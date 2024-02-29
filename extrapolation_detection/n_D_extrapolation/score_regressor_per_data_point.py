import itertools
from typing import Callable

import numpy as np
import pandas as pd

from core.s3_model_tuning.models.abstract_model import AbstractMLModel
from extrapolation_detection.machine_learning_util import data_handling as dh
from core.util.data_handling import split_target_features
from core.exploration_quantification.exploration_quantification import ArtificialPointGenerator


def score_per_sample(
    regressor: AbstractMLModel, x: pd.DataFrame, y: pd.DataFrame, metric: str = "mae"
) -> np.ndarray:
    """Tests the ann on given dataset and returns the score per datapoint"""

    assert metric in ["mse", "mae", "me", "rmse"]

    df = pd.DataFrame()

    df["y_real"] = y

    df["y_pred"] = regressor.predict(x)

    if metric == "mae":
        df["error"] = abs(df["y_pred"] - df["y_real"])
    elif metric == "me":
        df["error"] = df["y_pred"] - df["y_real"]
    elif metric == "mse" or metric == "rmse":
        df["error"] = (df["y_pred"] - df["y_real"]) ** 2
    else:
        raise ValueError("Please select a proper metric.")

    return df


def score_meshgrid(
    regressor: AbstractMLModel,
    system_simulation: Callable,
    x_tot: pd.DataFrame,
    mesh_points_per_axis: int = 100,
    metric: str = "mae",
) -> dict:
    """Calculates the error on a meshgrid for plotting purposes"""


    df_grid_scores = score_per_sample(regressor, x_grid, y_grid, metric)

    return xy_grid, df_grid_scores

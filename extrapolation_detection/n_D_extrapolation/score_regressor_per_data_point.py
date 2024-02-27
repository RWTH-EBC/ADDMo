import itertools
from typing import Callable

import numpy as np
import pandas as pd

from core.s3_model_tuning.models.abstract_model import AbstractMLModel
from extrapolation_detection.machine_learning_util import data_handling as dh


def score_per_sample(
    regressor: AbstractMLModel, x: np.ndarray, y: np.ndarray, metric: str = "mae"
) -> np.ndarray:
    """Tests the ann on given dataset and returns the score per datapoint"""

    assert metric in ["mse", "mae", "me", "rmse"]

    df = pd.DataFrame()

    df["y_real"] = y.squeeze()

    df["y_pred"] = regressor.predict(x)

    if metric == "mae":
        df["error"] = abs(df["y_pred"] - df["y_real"])
    elif metric == "me":
        df["error"] = df["y_pred"] - df["y_real"]
    elif metric == "mse" or metric == "rmse":
        df["error"] = (df["y_pred"] - df["y_real"]) ** 2
    else:
        raise ValueError("Please select a proper metric.")

    return df["error"]


def score_train_val_test(
    regressor: AbstractMLModel, available_data, metric: str = "mae"
) -> dict:
    """Scores the regressor on training, validation and test data"""
    x_train = available_data["x_train"]
    y_train = available_data["y_train"]
    x_val = available_data["x_val"]
    y_val = available_data["y_val"]
    x_test = available_data["x_test"]
    y_test = available_data["y_test"]

    errors = dict()
    errors["train_error"] = score_per_sample(regressor, x_train, y_train, metric)
    errors["val_error"] = score_per_sample(regressor, x_val, y_val, metric)
    errors["test_error"] = score_per_sample(regressor, x_test, y_test, metric)
    return errors


def score_remaining_data(
    regressor: AbstractMLModel, remaining_data, metric: str = "mae"
) -> dict:
    """Scores the regressor on remaining data"""
    x_remaining = remaining_data["x_remaining"]
    y_remaining = remaining_data["y_remaining"]

    errors = dict()
    errors["data_error"] = score_per_sample(regressor, x_remaining, y_remaining, metric)
    return errors


def score_meshgrid(
    regressor: AbstractMLModel,
    xy_tot_splitted,
    system_simulation: Callable,
    mesh_points_per_axis: int = 100,
    metric: str = "mae",
) -> dict:
    # Get bounds of nD plot
    bounds = []
    for i in range(xy_tot_splitted["available_data"]["x_train"].shape[1]):
        full_range = np.concatenate(
            (
                xy_tot_splitted["available_data"]["x_train"],
                xy_tot_splitted["available_data"]["x_val"],
                xy_tot_splitted["available_data"]["x_test"],
                xy_tot_splitted["non_available_data"]["x_remaining"],
            )
        )[:, i]
        min_val = np.amin(full_range)
        max_val = np.amax(full_range)
        bounds.append((min_val, max_val))

    # Generate Meshgrid
    spaces = [np.linspace(bound[0], bound[1], mesh_points_per_axis) for bound in bounds]
    mesh = np.meshgrid(*spaces)

    score_meshgrid_dct = dict()
    for i, space in enumerate(spaces):
        score_meshgrid_dct[xy_tot_splitted["header"][i]] = space

    # Initialize error array
    error_on_mesh = np.zeros([mesh_points_per_axis] * len(spaces))

    # Evaluate meshgrid with ANN and underlying model
    for index in itertools.product(*[range(mesh_points_per_axis)] * len(spaces)):
        point = [mesh[i][index] for i in range(len(spaces))]
        error_on_mesh[index] = score_per_sample(
            regressor, np.array(point).reshape(1, -1), system_simulation(*point), metric
        )

    score_meshgrid_dct["error_on_mesh"] = error_on_mesh

    return score_meshgrid_dct

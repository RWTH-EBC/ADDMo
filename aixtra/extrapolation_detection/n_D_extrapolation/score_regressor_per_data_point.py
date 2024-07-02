import numpy as np
import pandas as pd

from addmo.s3_model_tuning.models.abstract_model import AbstractMLModel


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


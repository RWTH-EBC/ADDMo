import warnings

import numpy as np
import pandas as pd

import tensorflow as tf

from addmo.s3_model_tuning.models.abstract_model import AbstractMLModel
from scikeras.wrappers import KerasRegressor

def calc_gradient(model:AbstractMLModel, x: pd.DataFrame) -> pd.DataFrame:
    '''Calculate the gradient of the model with respect to the input x.'''

    # check if its Keras model
    model = model.regressor
    if isinstance(model, tf.keras.Model):
        model = model
    elif isinstance(model, KerasRegressor):
        model = model.model_
    else:
        warnings.warn("Gradient calculation only for keras models. Returning zeros.")
        return pd.DataFrame(0, index=[0], columns=x.columns)

    x_ = tf.Variable(x)
    # Calculate the gradient
    with tf.GradientTape() as tape:
        tape.watch(x_)
        y = model(x_)
    gradient = tape.gradient(y, x_)

    # convert to pandas dataframe
    gradient = pd.DataFrame(gradient.numpy(), columns=x.columns)

    return gradient


def classify_gradient(gradients: pd.DataFrame, gradient_zero_margin: float = 1e-6):
    '''
    Classify gradient values into three categories:
    -1: negative gradient
    0: gradient close to zero (within +/- gradient_zero_margin)
    1: positive gradient

    Parameters:
    gradients (pd.Series): Series of gradient values
    gradient_zero_margin (float): Threshold for considering a gradient as zero.

    Returns:
    pd.Series: Classified gradients
    '''

    def classify(x):
        if abs(x) < gradient_zero_margin:
            return 0
        elif x > 0:
            return 1
        else:
            return -1

    return gradients.map(classify)

def calc_gradient_coverage(gradients_clf: pd.Series) -> pd.DataFrame:
    '''
    Calculate the coverage of the gradients_clf values.

    Parameters:
    gradients_clf (pd.Series): Series of gradients_clf values

    Returns:
    pd.DataFrame: DataFrame with the coverage of the gradients_clf values
    '''

    coverage = pd.Series()
    counts = gradients_clf.value_counts(normalize=True) * 100

    coverage.loc["Constant"] = counts.get(0, 0)
    coverage.loc["Increasing"] = counts.get(1, 0)
    coverage.loc["Decreasing"] = counts.get(-1, 0)

    return coverage
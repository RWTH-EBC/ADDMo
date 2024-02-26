import pandas as pd

from core.s2_data_tuning import feature_constructor
from core.util.data_handling import split_target_features
from core.s3_model_tuning.scoring.validator_factory import ValidatorFactory
from core.s3_model_tuning.models.model_factory import ModelFactory
from core.s3_model_tuning.scoring.abstract_scorer import ValidationScoring
from core.s3_model_tuning.models.abstract_model import AbstractMLModel
from core.s1_data_tuning_auto.config.data_tuning_auto_config import DataTuningAutoSetup
from core.s3_model_tuning.model_tuner import ModelTuner


def create_difference(config: DataTuningAutoSetup, xy):
    x_created = pd.DataFrame()

    for var_name in xy.columns:
        if var_name != config.name_of_target:
            series = feature_constructor.create_diff(xy[var_name])
            x_created[series.name] = series

    return x_created


def manual_target_lags(config: DataTuningAutoSetup, xy):
    # target_lags in format [first lag (int), second lag (int)]
    x_created = pd.DataFrame()

    for lag in config.target_lag:
        series = feature_constructor.create_lag(xy[config.name_of_target], lag)
        x_created[series.name] = series

    return x_created


def automatic_timeseries_target_lag_constructor(config: DataTuningAutoSetup, xy):
    x_created = pd.DataFrame()

    tuner = ModelTuner(config)

    # prepare data
    x, y = split_target_features(config.name_of_target, xy)

    model = tuner.tune_model(config.model, x, y)

    old_score = tuner.scorer.score_validation(model, x, y)

    # loop through to create lags as long as they improve the result
    for i in range(config.minimum_target_lag, len(x)):
        series = feature_constructor.create_lag(y, i)
        x_processed = pd.concat([x, series], axis=1, join="inner").bfill()

        new_model = tuner.tune_model(config.model, x_processed, y)
        new_score = tuner.scorer.score_validation(new_model, x_processed, y)

        if new_score <= old_score + config.min_increase_4_wrapper:
            break
        else:
            x_created[series.name] = series
            old_score = new_score

    return x_created


def manual_feature_lags(config: DataTuningAutoSetup, xy):
    # feature_lags in format {var_name: [first lag (int), second lag (int)]}

    x_created = pd.DataFrame()

    for var_name, lags in config.feature_lags.items():
        if var_name != config.name_of_target:
            for lag in lags:
                series = feature_constructor.create_lag(xy[var_name], lag)
                x_created[series.name] = series

    return x_created


def automatic_feature_lag_constructor(config: DataTuningAutoSetup, xy):
    x_created = pd.DataFrame()

    tuner = ModelTuner(config)

    # prepare data
    x, y = split_target_features(config.name_of_target, xy)

    model = tuner.tune_model(config.model, x, y)

    old_score = tuner.scorer.score_validation(model, x, y)

    # Loop through to create feature lags as long as they improve the result
    for column in x:
        temp_score = old_score
        for i in range(config.minimum_feature_lag, config.maximum_feature_lag + 1):
            series = feature_constructor.create_lag(x[column], i)
            x_processed = pd.concat([x, series], axis=1, join="inner")

            new_model = tuner.tune_model(config.model, x_processed, y)
            new_score = tuner.scorer.score_validation(new_model, x_processed, y)

            # choose the best lag for that feature
            if new_score > temp_score:
                temp_score = new_score
                x_best_lag = series

        # add best lag to feature space if good enough
        if temp_score >= old_score + config.min_increase_4_wrapper:
            x_created[x_best_lag.name] = x_best_lag

    return x_created

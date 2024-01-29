import pandas as pd
from sklearn.model_selection import train_test_split

from core.data_tuning import feature_constructor
from core.util.data_handling import split_target_features
from core.data_tuning.wrapper_model import DataTunerWrapperModel

from core.data_tuning_auto.config.data_tuning_auto_config import DataTuningAutoSetup

def create_difference(config: DataTuningAutoSetup, data):

    x_created = pd.DataFrame()

    for var_name in data.columns:
        if var_name != config.name_of_target
            series = feature_constructor.create_difference(data[var_name])
            x_created[series.name] = series

    return x_created

def manual_target_lags(config: DataTuningAutoSetup, xy):
    # target_lags in format [first lag (int), second lag (int)]
    x_created = pd.DataFrame()

    for lag in config.name_of_target:
        series = feature_constructor.create_lag(xy[config.name_of_target], lag)
        x_created[series.name] = series

    return x_created

def automatic_timeseries_target_lag_constructor(config: DataTuningAutoSetup, xy):
    x_created = pd.DataFrame()

    Wrapper = DataTunerWrapperModel(config)

    # prepare data
    x, y = split_target_features(config.name_of_target, xy)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25) #todo: überflüssig
    # da alle genutzt werden darf?

    old_score = Wrapper.train_score(x_train, x_test, y_train, y_test)

    # loop through to create lags as long as they improve the result
    for i in range(config.minimum_target_lag, len(x_train)):
        series = feature_constructor.create_lag(y, i)
        x_processed = pd.concat([x, series], axis=1, join="inner")

        # redo identical splitting
        x_train_processed = x_processed.loc[x_train.index]
        x_test_processed = x_processed.loc[x_test.index]

        new_score = Wrapper.train_score(x_train_processed, x_test_processed, y_train,
                                        y_test)

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

def automatic_feature_lag_constructor(config: DataTuningAutoSetup, data):
    x_created = pd.DataFrame()

    Wrapper = DataTunerWrapperModel(config)

    x, y = split_target_features(config.name_of_target, data)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    old_score = Wrapper.train_score(x_train, x_test, y_train, y_test)

    # Loop through to create feature lags as long as they improve the result
    for column in x:
        temp_score = old_score
        for i in range(config.minimum_feature_lag, config.maximum_feature_lag + 1):
            series = feature_constructor.create_lag(x[column], i)
            x_processed = pd.concat([x, series], axis=1, join="inner")

            x_train_processed = x_processed.loc[x_train.index]
            x_test_processed = x_processed.loc[x_test.index]

            new_score = Wrapper.train_score(x_train_processed, x_test_processed,
                                            y_train,
                                            y_test)
            # choose the best lag for that feature
            if new_score > temp_score:
                temp_score = new_score
                x_best_lag = series

        # add best lag to feature space if good enough
        if temp_score >= old_score + config.min_increase_4_wrapper:
            x_created[x_best_lag.name] = x_best_lag

    return x_created


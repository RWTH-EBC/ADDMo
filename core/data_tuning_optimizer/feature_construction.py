import pandas as pd
from sklearn.model_selection import train_test_split

from core.data_tuning import feature_constructor
from core.util.data_handling import split_target_features
from core.data_tuning.wrapper_model import DataTunerWrapperModel

def manual_feature_lags(config_gui, xy):
    # feature_lags in format {var_name: [first lag (int), second lag (int)]}

    x_created = pd.DataFrame()

    for var_name, lags in config_gui["feature_lags"].items():
        if var_name != config_gui["target"]:
            for lag in lags:
                series = feature_constructor.create_lag(xy[var_name], lag)
                x_created[series.name] = series

    return x_created

def manual_target_lags(config_gui, xy):
    # target_lags in format [first lag (int), second lag (int)]
    x_created = pd.DataFrame()

    for lag in config_gui["target_lags"]:
        series = feature_constructor.create_lag(xy[config_gui["target"]], lag)
        x_created[series.name] = series

    return x_created

def automatic_timeseries_target_lag_constructor(config_gui, xy):
    x_created = pd.DataFrame()

    Wrapper = DataTunerWrapperModel(config_gui)

    # prepare data
    x, y = split_target_features(config_gui.name_of_target, xy)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    old_score = Wrapper.train_score(x_train, x_test, y_train, y_test)

    # loop through to create lags as long as they improve the result
    for i in range(config_gui.minimum_target_lag, len(x_train)):
        series = feature_constructor.create_lag(y, i)
        x_processed = pd.concat([x, series], axis=1, join="inner")

        # redo identical splitting
        x_train_processed = x_processed.loc[x_train.index]
        x_test_processed = x_processed.loc[x_test.index]

        new_score = Wrapper.train_score(x_train_processed, x_test_processed, y_train,
                                        y_test)

        if new_score <= old_score + config_gui["min_increase"]:
            break
        else:
            x_created[series.name] = series
            old_score = new_score

    return x_created

def automatic_feature_lag_constructor(config_gui, data):
    x_created = pd.DataFrame()

    Wrapper = DataTunerWrapperModel(config_gui)

    x, y = split_target_features(config_gui.name_of_target, data)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    old_score = Wrapper.train_score(x_train, x_test, y_train, y_test)

    # Loop through to create feature lags as long as they improve the result
    for column in x:
        temp_score = old_score
        for i in range(config_gui["min_feature_lag"], config_gui["max_feature_lag"] + 1):
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
        if temp_score >= old_score + config_gui["min_increase"]:
            x_created[x_best_lag.name] = x_best_lag

    return x_created

def create_difference(config_gui, data):

    x_created = pd.DataFrame()

    for var_name in data.columns:
        if var_name != config_gui["target"]:
            series = feature_constructor.create_difference(data[var_name])
            x_created[series.name] = series

    return x_created

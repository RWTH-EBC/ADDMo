import pandas as pd

from core.s1_data_tuning_auto.config.data_tuning_auto_config import DataTuningAutoSetup
from core.util.load_save import load_data
from core.s1_data_tuning_auto import feature_construction as fc
from core.s1_data_tuning_auto import feature_selection as fs
from core.util.data_handling import split_target_features
from core.util.experiment_logger import ExperimentLogger
class DataTunerAuto:
    """
    This class is used to run the data tuning process automatically, possibly leading to
    different results depending on the run.
    """

    def __init__(self, config: DataTuningAutoSetup):
        self.config = config
        self.xy_raw = load_data(self.config.abs_path_to_data)
        self.x, self.y = split_target_features(self.config.name_of_target, self.xy_raw)

    def tune_auto(self):
        """
        This method is used to run the data tuning process automatically, possibly leading to
        different results depending on the run.
        """
        self.feature_construction()
        self.feature_selection()
        return self.x

    def feature_construction(self):
        """
        This method is used to construct new features from the raw data.
        """
        if self.config.create_differences:
            x_created = fc.create_difference(self.config, self.xy_raw)
            self.x = pd.concat([self.x, x_created], axis=1, join="inner").bfill()
        if self.config.create_manual_target_lag:
            x_created = fc.manual_target_lags(self.config, self.xy_raw)
            self.x = pd.concat([self.x, x_created], axis=1, join="inner").bfill()
        if self.config.create_automatic_timeseries_target_lag:
            x_created = fc.automatic_timeseries_target_lag_constructor(self.config, self.xy_raw)
            self.x = pd.concat([self.x, x_created], axis=1, join="inner").bfill()
        if self.config.create_manual_feature_lags:
            x_created = fc.manual_feature_lags(self.config, self.xy_raw)
            self.x = pd.concat([self.x, x_created], axis=1, join="inner").bfill()
        if self.config.create_automatic_feature_lags:
            x_created = fc.automatic_feature_lag_constructor(self.config, self.xy_raw)
            self.x = pd.concat([self.x, x_created], axis=1, join="inner").bfill()


    def feature_selection(self):
        if self.config.manual_feature_selection:
            self.x = fs.manual_feature_select(self.config, self.x)
        if self.config.filter_low_variance:
            self.x = fs.filter_low_variance(self.config, self.x)
        if self.config.filter_ICA:
            self.x = fs.filter_ica(self.config, self.x)
        if self.config.filter_univariate:
            self.x = fs.filter_univariate(self.config, self.x, self.y)
        if self.config.filter_recursive_embedded:
            self.x = fs.recursive_feature_selection_embedded(self.config, self.x)
        if self.config.filter_recursive_embedded:
            self.x = fs.embedded__recursive_feature_selection(self.config, self.x)
        if self.config.wrapper_sequential_feature_selection:
            self.x = fs.recursive_feature_selection_wrapper_scikit_learn(self.config, self.x,
                                                                         self.y)


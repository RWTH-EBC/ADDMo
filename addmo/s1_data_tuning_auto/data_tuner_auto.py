import pandas as pd
from addmo.s1_data_tuning_auto.config.data_tuning_auto_config import DataTuningAutoSetup
from addmo.util.load_save import load_data
from addmo.s1_data_tuning_auto import feature_construction as fc
from addmo.s1_data_tuning_auto import feature_selection as fs
from addmo.util.data_handling import split_target_features

class DataTunerAuto:
    """
    This class is used to run the system_data tuning process automatically, possibly leading to
    different results depending on the run.
    """

    def __init__(self, config: DataTuningAutoSetup):
        self.config = config
        self.xy_raw = load_data(self.config.abs_path_to_data)
        self.x, self.y = split_target_features(self.config.name_of_target, self.xy_raw)

    def tune_auto(self):
        """
        This method is used to run the system_data tuning process automatically, possibly leading to
        different results depending on the run.
        """
        self.feature_construction()
        self.feature_selection()
        return self.x

    def feature_construction(self):
        """
        This method is used to construct new features from the raw system_data.
        """
        if self.config.create_differences:
            x_created = fc.create_difference(self.config, self.xy_raw)
            self.x = pd.concat([self.x, x_created], axis=1, join="inner").bfill()
        if self.config.create_manual_target_lag:
            x_created = fc.manual_target_lags(self.config, self.xy_raw)
            self.x = pd.concat([self.x, x_created], axis=1, join="inner").bfill()
        if self.config.create_manual_feature_lags:
            x_created = fc.manual_feature_lags(self.config, self.xy_raw)
            self.x = pd.concat([self.x, x_created], axis=1, join="inner").bfill()



    def feature_selection(self):
        """
        This method is used to select features from the raw system_data.
        """
        if self.config.manual_feature_selection:
            self.x = fs.manual_feature_select(self.config, self.x)

        if self.config._filter_recursive_by_count:
            self.x = fs.recursive_feature_selection_by_count(self.config, self.x, self.y)

        if self.config._filter_recursive_by_score:
            self.x = fs.recursive_feature_selection_by_score(self.config, self.x, self.y)


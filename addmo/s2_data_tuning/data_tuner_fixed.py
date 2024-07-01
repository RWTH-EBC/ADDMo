import pandas as pd

from addmo.s2_data_tuning.config.data_tuning_config import DataTuningFixedConfig
from addmo.s2_data_tuning import feature_constructor as fc
from addmo.util.load_save import load_data
from addmo.util.experiment_logger import ExperimentLogger


class DataTunerByConfig:
    """Tunes the system_data in a fixed manner. Without randomness."""
    def __init__(self, config: DataTuningFixedConfig):
        self.config = config


    def update_x_raw(self, x_sample: pd.DataFrame):
        """
        Update the x_processed DataFrame with new system_data.
        E.g. for online environments or recursive predictions. #todo: recursive

        The input DataFrame must have a DateTimeIndex in equal resolution.
        It can contain features and/or target values. This method either overwrites
        existing values or appends new system_data depending on the index match.
        """
        # Overwrite existing system_data or append new system_data
        self.xy_raw = self.xy_raw.combine_first(x_sample)

        # limit the maximum size length of the df to 100 lines
        self.xy_raw = self.xy_raw.tail(100)

    def update_y(self, y_sample: pd.DataFrame): #Todo: notwenig?
        """recursive prediction"""
        # Overwrite existing system_data or append new system_data
        self.xy_raw = self.xy_raw.combine_first(y_sample)


    def tune_fixed(self, xy_raw):
        x_processed = pd.DataFrame(index=xy_raw.index)
        for feature_name in self.config.features:
            # extract feature name and modification type
            if '___' in feature_name:
                original_name, modification = feature_name.split('___')
                var = xy_raw[original_name]


                if modification.startswith('lag'):
                    lag = int(modification[3:])
                    series = fc.create_lag(var, lag)
                else:
                    # get the other methods dynamically from module
                    method = getattr(fc, "create_" + modification)
                    series = method(var)
                x_processed[series.name] = series

            # keep desired raw features
            elif feature_name in xy_raw.columns:
                x_processed[feature_name] = xy_raw[feature_name]

            else:
                print(f"Feature <{feature_name}> not present in loaded system_data.")

        return x_processed
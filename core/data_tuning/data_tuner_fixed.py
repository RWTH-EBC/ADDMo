import pandas as pd

from core.data_tuning.config.data_tuning_config import DataTuningFixedConfig
from core.data_tuning import feature_constructor as fc
from core.data_tuning.data_importer import load_data
from core.util.experiment_logger import ExperimentLogger

class DataTunerByConfig():
    '''Tunes the data in a fixed manner. Without randomness.'''
    def __init__(self, config: DataTuningFixedConfig, logger:ExperimentLogger):
        self.config = config
        self.logger = logger
        self.xy_raw = load_data(self.config.path_to_raw_data)
        self.x_processed = pd.DataFrame(index=self.xy_raw.index)  # Initialize with the same index

    def update_x_raw(self, x_sample: pd.DataFrame):
        """
        Update the x_processed DataFrame with new data.
        E.g. for online environments or recursive predictions. #todo: recursive

        The input DataFrame must have a DateTimeIndex in equal resolution.
        It can contain features and/or target values. This method either overwrites
        existing values or appends new data depending on the index match.
        """
        # Overwrite existing data or append new data
        self.xy_raw = self.xy_raw.combine_first(x_sample)

        # limit the maximum size length of the df to 100 lines
        self.xy_raw = self.xy_raw.tail(100)

    def update_y(self, y_sample: pd.DataFrame): #Todo: notwenig?
        '''recursive prediction'''
        # Overwrite existing data or append new data
        self.xy_raw = self.xy_raw.combine_first(y_sample)


    def tune_fixed(self):
        for feature_name in self.config.features:
            # extract feature name and modification type
            if '___' in feature_name:
                original_name, modification = feature_name.split('___')
                var = self.xy_raw[original_name]

                if modification.startswith('lag'):
                    lag = int(modification[3:])
                    series = fc.create_lag(var, lag)
                    self.x_processed[series.name] = series

                elif modification == 'diff':
                    series = fc.create_difference(var)
                    self.x_processed[series.name] = series

                elif modification == 'squared':
                    series = fc.create_squared(var)
                    self.x_processed[series.name] = series

            # keep desired raw features
            elif feature_name in self.xy_raw.columns:
                self.x_processed[feature_name] = self.xy_raw[feature_name]

            else:
                print(f"Feature <{feature_name}> not present in loaded data.")

        return self.x_processed
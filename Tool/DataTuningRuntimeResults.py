import os

import pandas as pd
from sklearn.externals import joblib
from core.data_tuning_optimizer.config.data_tuning_config import DataTuningSetup


class DataTuningRuntimeResults:
    """
    Object that stores all the runtime results of Data Tuning

    Stores all the data frames generated after each of the following processes and their execution times:
    1. Import Data
    2. Preprocessing
    3. Period Selection
    4. Feature Construction
    5. Feature Selection

    The 'Scaler object' is also stored dynamically in the Preprocessing step (check Preprocessing.py)
    """

    def __init__(self, **kwargs):

        self.df_import_data = pd.DataFrame
        self.df_preprocessing_data = pd.DataFrame
        self.df_period_selection_data = pd.DataFrame
        self.df_feature_construction_data = pd.DataFrame
        self.df_feature_selection_data = pd.DataFrame

        self.import_time = 0
        self.preprocessing_time = 0
        self.period_selection_time = 0
        self.feature_construction_time = 0
        self.feature_selection_time = 0
        self.total_time = 0

    def store_results(self, DT_Setup_object: DataTuningSetup):
        print(
            "Saving Data Tuning Runtime Results class Object as a pickle in path: \n'%s'"
            % os.path.join(
                DT_Setup_object.abs_path_to_result_folder, "DataTuningRuntimeResults.save"
            )
        )

        # Save the object as a pickle for reuse
        joblib.dump(
            self,
            os.path.join(
                DT_Setup_object.abs_path_to_result_folder, "DataTuningRuntimeResults.save"
            ),
        )

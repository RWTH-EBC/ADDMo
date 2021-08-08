import os

import pandas as pd
from sklearn.externals import joblib

import SharedVariables as SV

class DataTuningRuntimeResults:

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

    def store_results(self):
        print("Saving Data Tuning Runtime Results class Object as a pickle in path: '%s'" % os.path.join(SV.ResultsFolder, "DataTuningRuntimeResults.save"))

        # Save the object as a pickle for reuse
        joblib.dump(self, os.path.join(SV.ResultsFolder, "DataTuningRuntimeResults.save"))
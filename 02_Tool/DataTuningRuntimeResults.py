
# DT and MT classes replace SV (DT class now almost replaces SV, todo: InputData + NameOfSignal + global variables in general ,decide their scope, scope will be  clear when MT is handled)
import os

import pandas as pd

import SharedVariables as SV

from sklearn.externals import joblib


class DataTuningRuntimeResults:

    def __init__(self, *args, **kwargs):

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
import os

import numpy as np
from sklearn.model_selection import TimeSeriesSplit

import SharedVariables as SV

from sklearn.externals import joblib
from sklearn.feature_selection import mutual_info_regression, f_regression


def set_data(DTS_object):

    print("Saving Data Tuning Setup class Object as a pickle in path: '%s'" % os.path.join(SV.ResultsFolder, "DataTuningSetup.save"))

    # Save the object as a pickle for reuse
    joblib.dump(DTS_object, os.path.join(SV.ResultsFolder, "DataTuningSetup.save"))

class DataTuningSetup:
    #****** first we need to set setup data from SV, coz GUI sets all variables in SV, if SV has to be eliminated even GUI file must be changed. ex: compute_DT() (done)
    # scaler and execution time are the runtime results (done)
    # Is it recommended to write a parameterized __init__ if there are so many variables (to set default values)? (done)
    # or should we go for the default constructor and assign hardcoded default values? (done)
    # use kwargs today (done, dont see any use of args or kwargs as we do not send anything to the object as a parameter) (done)

    def __init__(self, *args, **kwargs):

        # Global Variables
        ####################################
        self.NameOfData = "AHU Data1"
        self.NameOfExperiment = "NoOL"
        self.NameOfSignal = "Empty"

        # Wrapper variables
        self.WrapperModel = SV.WrapperModels  # should this class be made independent of SV, if yes, why?
        self.WrapperParams = SV.WrapperParams  # should this class be made independent of SV, if yes, why?
        self.MinIncrease = 0.005

        self.ExecutionTime = 0

        # ImportData Variables
        #############################################################

        # Preprocessing variables
        #############################################################
        self.NaNDealing = "bfill"
        self.InitManFeatureSelect = False
        self.InitFeatures = [1, 2, 3, 8, 9]
        self.StandardScaling = False
        self.RobustScaling = True
        self.NoScaling = False
        self.Resample = False
        self.Resolution = "60min"                    # e.g. "60min" means into buckets of 60minutes, "30s" to seconds
        self.WayOfResampling = [np.mean, np.mean, np.mean]
        # e.g. for a 3 column data set(index not counted):[np.sum,np.mean, np.mean] first column will be summed up
        # all other will be meant Define way of resampling per column, available: Resample to larger interval:
        # np.sum, np.mean, np.median or a redefined aggregation method


        # PeriodSelection Variables
        #############################################################
        self.ManSelect = False
        self.StartDate = '2016-06-02 00:00'
        self.EndDate = '2016-06-16 00:00'
        self.TimeSeriesPlot = False

        # FeatureConstruction Variables
        #############################################################
        self.Cross_auto_cloud_correlation_plotting = False
        self.LagsToBePlotted = 200
        self.DifferenceCreate = False
        self.FeaturesDifference = True
        self.ManOwnlagCreate = False
        self.OwnLag = [1]
        self.ManFeaturelagCreate = False
        self.FeatureLag = [[3, 2], [], [], [], [], [], [], [], [1], []]
        self.AutomaticTimeSeriesOwnlagConstruct = False
        self.MinOwnLag = 1
        self.AutoFeaturelagCreate = False
        self.MinFeatureLag = 1
        self.MaxFeatureLag = 20

        # FeatureSelection Variables
        #############################################################
        self.ManFeatureSelect = False
        self.FeatureSelect = [2, 3, 6, 8, 9]
        self.LowVarianceFilter = False
        self.Threshold_LowVarianceFilter = 0.1
        self.ICA = False
        self.UnivariateFilter = False
        self.Score_func = mutual_info_regression  # <mutual_info_regression> or <f_regression>
        self.SearchMode = "percentile"
        self.Param_univariate_filter = 50
        self.EstimatorEmbedded = SV.rf  # should this class be made independent of SV, if yes, why?
        self.N_feature_to_select_RFE = 18
        self.CV_DT = TimeSeriesSplit(n_splits=3)
        self.Threshold_embedded = "median"

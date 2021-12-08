
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import mutual_info_regression, f_regression
from math import log

from BlackBoxes import *

import SharedVariablesFunctions as SVF

class DataTuningSetup:
    """
    Object that stores all the setup and user input information of Data Tuning

    # -----------------------Global Variables-------------------------------

    NameOfData = "AHU Data1"  # Set name of the folder where the experiments shall be saved, e.g. the name of the observed data
    NameOfExperiment = "NoOL"  # Set name of the experiments series
    NameOfSignal = "Empty" # Name of the target variable
    RootDir = "Empty" # The root directory of the file
    PathToData = "Empty"  # Path to the file that has the data
    ResultsFolder = "Empty" # Path to the folder where the results will be stored
    PathToPickles = "Empty" # Path to the folder where the pickles will be stored
    ColumnOfSignal = 1

    # -----------------------Wrapper Variables-------------------------------

    <!-- Defined in SharedVariables.py
    Hyperparametergrids = {"ANN":hp.choice("number_of_layers",
                            [
                            {"1layer": scope.int(hp.qloguniform("1.1", log(1), log(1000), 1))},
                            {"2layer": [scope.int(hp.qloguniform("1.2", log(1), log(1000), 1)), scope.int(hp.qloguniform("2.2", log(1), log(1000), 1))]},
                            {"3layer": [scope.int(hp.qloguniform("1.3", log(1), log(1000), 1)), scope.int(hp.qloguniform("2.3", log(1), log(1000), 1)), scope.int(hp.qloguniform("3.3", log(1), log(1000), 1))]}
                            ]),
                           "SVR":{"C": hp.loguniform("C", log(1e-4), log(1e4)), "gamma": hp.loguniform("gamma", log(1e-3), log(1e4)), "epsilon": hp.loguniform("epsilon", log(1e-4), log(1))},
                           "GB":{"n_estimators": scope.int(hp.qloguniform("n_estimators", log(1), log(1e3), 1)), "max_depth": scope.int(hp.qloguniform("max_depth", log(1),log(100), 1)), "learning_rate":hp.loguniform("learning_rate", log(1e-2), log(1)), "loss":hp.choice("loss",["ls", "lad", "huber", "quantile"])},
                           "Lasso":{"alpha": hp.loguniform("alpha", log(1e-10), log(1e6))},
                           "RF":None}
    WrapperModels = {"ANN":ann_bayesian_predictor,"GB":gradientboost_bayesian,"Lasso":lasso_bayesian,"SVR":svr_bayesian_predictor,"RF":rf_predictor}
    -->

    EstimatorWrapper = SVF.WrapperModels["RF"]  # state one blackbox model from "BlackBoxes.py", without parenthesis, e.g. <rf_predictor>
    WrapperParams = [SVF.Hyperparametergrids["RF"], None, None, False]  # state the parameters that the model should have . Eg. [None, None, None, False] or [HyperparameterGrid, TimeSeriesSplit(n_splits=3), 30, False]

    # 1st entry = hyperparametergrid
    # 2nd= crossvalidation
    # 3rd= max_eval
    # 4th= recursive (consider turning on if creating ownlags via wrapper; makes rf about 4 times slower, to other models just little influence)
    MinIncrease = 0.005  # minimum difference between two scores(Score-error) to accept a change to the original data, e.g. NoOwnlag + MinIncrease < 1Ownlag in order to add a new ownlag

    # -----------------------ImportData Variables-------------------------------

    # -----------------------Preprocessing Variables-------------------------------

    # Define if data should be resampled
    Resample = False
    # If Resample is True the following resolution is required
    Resolution = "60min"  # e.g. "60min" means into buckets of 60minutes, "30s" to seconds
    WayOfResampling = [np.mean, np.mean,
                       np.mean]  # e.g. for a 3 column data set(index not counted):[np.sum, np.mean, np.mean] first column will be summed up all other will be meaned
    # Define way of resampling per column, available: Resample to larger interval: np.sum, np.mean, np.median or a selfdefined aggregation method

    # Initial manual feature selection
    # Manual selection of Features by their Column number
    InitManFeatureSelect = False
    InitFeatures = [1, 2, 3, 8, 9]  # e.g.[2, 3] #enter the column number of the features you want to keep(start to count from 0 with first Column after Index, Column of signal needs to be counted, but will be kept in any case)

    # Define how NaN values should be handled
    NaNDealing = "bfill"  # possible "bfill", "ffill", "dropna" or "None" #advised: bfill or ffill

    # Define how data should be scaled and centered, preferenced: by experience: RobustScaling: Reason:(AHU1 Testing)
    StandardScaling = False  # use if your data does not contain outliers
    # or
    RobustScaling = True  # use if your data contains outliers
    #
    NoScaling = False  # use only if your data is already scaled, centered etc. or you do tests!

    # -----------------------PeriodSelection Variables--------------------------

    # Time Series Plot
    TimeSeriesPlot = False

    # Manual Period Selection
    ManSelect = False
    StartDate = '2016-06-02 00:00'  # start day of data set
    EndDate = '2016-06-16 00:00'  # end day of data set

    # -----------------------FeatureConstruction Variables----------------------

    # Production of Cross and Autocorrelation Plots in order to find meaningful OwnLags and FeatureLags
    Cross_auto_cloud_correlation_plotting = False
    LagsToBePlotted = 200  # number of lags which will be plotted in x-axis

    # Feature difference creation (building the derivatie of the features)
    DifferenceCreate = False
    FeaturesDifference = True  # "True" if a derivative should be created for all features; [2, 3, 5, 10] for certain features, #enter the column number of the features you want to keep(start to count from 0 with first Column after Index, don´t count Column of signal)

    # Manual FeatureLag construction
    ManFeaturelagCreate = False
    FeatureLag = [[3, 2], [], [], [], [], [], [], [], [1],
                  []]  # e.g. for a data with 6 columns:[[1],[1,2],[],[24],[],[]];type in an array of lags for each feature(signal column included), starting with first array = first column; for the column of signal put in anything, e.g. [0], it won´t be used anyways; through DifferenceCreate created features don´t need to be counted any more

    # Automatic FeatureLag construction (via wrapper); adds the one best featurelag per feature
    AutoFeaturelagCreate = False
    MinFeatureLag = 1
    MaxFeatureLag = 20

    # Manual Ownlag construction
    # 1 lag is as long as the resolution of your preprocessed data
    ManOwnlagCreate = False
    OwnLag = [1]  # [96*7] #[1,2,3,4, 24, 48] type in the ownlags

    # Automatic construction of time series OwnLags (via wrapper) counting up from MinOwnLag, stopping if an additional Ownlag reduces the score, this method is conducted as LAST METHOD in feature selection. Take care not to add OwnLags by yourself which are in the same range as AutomaticOwnlagSelect, since they will affect the score.
    AutomaticTimeSeriesOwnlagConstruct = False
    MinOwnLag = 1  # minimal OwnLag which shall be considered; must be an integer; 0 would be the signal itself(not meaningful)

    # -----------------------FeatureSelection Variables-------------------------

    # Manual selection of Features by their Column number (After Feature Construction)
    ManFeatureSelect = False
    FeatureSelect = [2, 3, 6, 8, 9]  # e.g.[2, 3] #enter the column number of the features you want to keep(start to count from 0 with first Column after Index, Column of signal needs to be counted, but will be kept in any case)(also created features must be selected here if they should be kept, except lags created through automatic_ownlag_constructor)

    # Removing features with low variance, e.g. for pre-filtering
    LowVarianceFilter = False
    Threshold_LowVarianceFilter = 0.1  # 0.1 (reasonable value) #removes all features with a lower variance than the stated threshold, variance is calculated with scaled data(if a scaler was used, regularly only features that are always the same have a small variance)

    # Filter: Independent Component Analysis(ICA)
    ICA = False  # ICA doesn´t seem advisable

    # Filter Univariate with scoring function "f_regression" or "mutual_info_regression" and search mode : {‘percentile’, ‘k_best’,(not working at the moment: ‘fpr’, ‘fdr’, ‘fwe’)}
    UnivariateFilter = False
    Score_func = mutual_info_regression  # <mutual_info_regression> or <f_regression>
    SearchMode = "percentile"
    Param_univariate_filter = 50  # if percentile: percent of features to keep; if Kbest: number of top features to keep; if fpr:The highest p-value for features to be kept; if FWE:The highest uncorrected p-value for features to keep

    # Embedded methods start-----------------
    # define and set Estimator which shall be used in all embedded Methods(only present in feature selection)
    rf = RandomForestRegressor(max_depth=10e17,
                               random_state=0)  # have to be defined so that they return "feature_importance", more implementation have to be developed
    lasso = linear_model.Lasso(max_iter=10e8)
    EstimatorEmbedded = rf  # e.g. <rf>; set one of the above models to be used

    # Embedded feature selection by setting an importance threshold (not recursive)
    EmbeddedFeatureSelectionThreshold = False
    Threshold_embedded = "median"  #from 0-1 or "median" or "mean"

    # Embedded feature selection (Recursive Feature Selection)
    RecursiveFeatureSelection = False
    # for a specific number of important features enter number here
    N_feature_to_select_RFE = 18  # enter an integer or "automatic" if optimal number shall be found automatic, only automatic supports Crossvalidation
    # if N_feature_to_select_RFE != "automatic" no crossvalidation is conducted even if CV_DT != 0
    CV_DT = TimeSeriesSplit(
        n_splits=3)  # set 0 if no CrossValidation while fitting is wished, also any kind of crossvalidater can be entered here
    # Embedded end --------------------

    # Wrapper recursive feature selection with the initially stated model for wrapper methods
    WrapperRecursiveFeatureSelection = False
    """

    def __init__(self, **kwargs):

        # -----------------------Global Variables-------------------------------

        self.NameOfData = "AHU Data1"
        self.NameOfExperiment = "NoOL"
        self.NameOfSignal = "Empty"
        self.RootDir = "Empty"
        self.PathToData = "Empty"
        self.ResultsFolder = "Empty"
        self.PathToPickles = "Empty"
        self.ColumnOfSignal = 1

        # -----------------------Wrapper Variables-------------------------------

        self.EstimatorWrapper = SVF.WrapperModels["RF"]
        self.WrapperParams = [SVF.Hyperparametergrids["RF"], None, None, False]
        self.MinIncrease = 0.005

        # -----------------------ImportData Variables-------------------------------

        # -----------------------Preprocessing Variables-------------------------------

        self.NaNDealing = "bfill"
        self.InitManFeatureSelect = False
        self.InitFeatures = [1, 2, 3, 8, 9]
        self.StandardScaling = False
        self.RobustScaling = True
        self.NoScaling = False
        self.Resample = False
        self.Resolution = "60min"
        self.WayOfResampling = [np.mean, np.mean, np.mean]

        # -----------------------PeriodSelection Variables-------------------------------

        self.ManSelect = False
        self.StartDate = '2016-06-02 00:00'
        self.EndDate = '2016-06-16 00:00'
        self.TimeSeriesPlot = False

        # -----------------------FeatureConstruction Variables-------------------------------

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

        # -----------------------FeatureSelection Variables-------------------------------

        self.ManFeatureSelect = False
        self.FeatureSelect = [2, 3, 6, 8, 9]
        self.LowVarianceFilter = False
        self.Threshold_LowVarianceFilter = 0.1
        self.ICA = False
        self.UnivariateFilter = False
        self.Score_func = mutual_info_regression
        self.SearchMode = "percentile"
        self.Param_univariate_filter = 50

        rf = RandomForestRegressor(max_depth=10e17, random_state=0)

        self.EstimatorEmbedded = rf
        self.RecursiveFeatureSelection = False
        self.N_feature_to_select_RFE = 18
        self.CV_DT = TimeSeriesSplit(n_splits=3)
        self.EmbeddedFeatureSelectionThreshold = False
        self.Threshold_embedded = "median"

        self.WrapperRecursiveFeatureSelection = False

    def dump_object(self):
        print("Saving Data Tuning Setup class Object as a pickle in path: '%s'" % os.path.join(self.ResultsFolder,
                                                                                               "DataTuningSetup.save"))
        # Save the object as a pickle for reuse
        joblib.dump(self, os.path.join(self.ResultsFolder, "DataTuningSetup.save"))

    def documentation_DataTuning(self, timestart, timeend):

        print("Documentation")
        # dump the name of signal in the resultsfolder, so that you can always be pulled whenever you want to come back to that specific "Final Input Data"
        joblib.dump(self.NameOfSignal, os.path.join(self.ResultsFolder, "NameOfSignal.save"))

        # saving the methodology of creating FinalInputData in the ExcelFile "Settings"
        DfMethodology = pd.DataFrame(index=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                                     columns=["GlobalVariables", "ImportData", "Preprocessing", "PeriodSelection",
                                              "FeatureConstruction", "FeatureSelection"])
        # adding information to the dataframe
        DfMethodology.at[1, "GlobalVariables"] = "NameOfData = %s" % self.NameOfData
        DfMethodology.at[2, "GlobalVariables"] = "NameOfExperiment = %s" % self.NameOfExperiment
        DfMethodology.at[3, "GlobalVariables"] = "NameOfSignal = %s" % self.NameOfSignal
        DfMethodology.at[4, "GlobalVariables"] = "Model used for Wrappers = %s" % self.EstimatorWrapper.__name__
        DfMethodology.at[5, "GlobalVariables"] = "Parameter used for Wrappers = %s" % self.WrapperParams
        DfMethodology.at[6, "GlobalVariables"] = "MinIncrease for Wrappers = %s" % self.MinIncrease
        DfMethodology.at[7, "GlobalVariables"] = "Pipeline took %s seconds" % (timeend - timestart)

        DfMethodology.at[1, "Preprocessing"] = "How to deal NaN´s = %s" % self.NaNDealing
        DfMethodology.at[2, "Preprocessing"] = "Initial feature select = %s" % self.InitManFeatureSelect
        if self.InitManFeatureSelect == True:
            DfMethodology.at[3, "Preprocessing"] = "Features selected = %s" % self.InitFeatures
        DfMethodology.at[4, "Preprocessing"] = "StandardScaler = %s" % self.StandardScaling
        DfMethodology.at[5, "Preprocessing"] = "RobustScaler = %s" % self.RobustScaling
        DfMethodology.at[6, "Preprocessing"] = "NoScaling = %s" % self.NoScaling
        DfMethodology.at[7, "Preprocessing"] = "Resample = %s" % self.Resample
        if self.Resample == True:
            DfMethodology.at[8, "Preprocessing"] = "Resolution = %s" % self.Resolution
            DfMethodology.at[9, "Preprocessing"] = "WayOfResampling = %s" % self.WayOfResampling

        DfMethodology.at[1, "PeriodSelection"] = "ManualSelection = %s" % self.ManSelect
        if self.ManSelect == True:
            DfMethodology.at[2, "PeriodSelection"] = "%s till %s" % (self.StartDate, self.EndDate)
        DfMethodology.at[3, "PeriodSelection"] = "TimeSeriesPlot = %s" % self.TimeSeriesPlot

        DfMethodology.at[
            1, "FeatureConstruction"] = "Cross, auto, cloud correlation plot= %s" % self.Cross_auto_cloud_correlation_plotting
        if self.Cross_auto_cloud_correlation_plotting == True:
            DfMethodology.at[2, "FeatureConstruction"] = "LagsToBePlotted= %s" % self.LagsToBePlotted
        DfMethodology.at[3, "FeatureConstruction"] = "DifferenceCreate= %s" % self.DifferenceCreate
        if self.DifferenceCreate == True:
            Word = "All" if self.FeaturesDifference == True else self.FeaturesDifference
            DfMethodology.at[4, "FeatureConstruction"] = "FeaturesToCreateDifference= %s" % Word
        DfMethodology.at[5, "FeatureConstruction"] = "Manual creation of OwnLags= %s" % self.ManOwnlagCreate
        if self.ManOwnlagCreate == True:
            DfMethodology.at[6, "FeatureConstruction"] = "OwnLags= %s" % self.OwnLag
        DfMethodology.at[7, "FeatureConstruction"] = "Manual creation of FeatureLags= %s" % self.ManFeaturelagCreate
        if self.ManFeaturelagCreate == True:
            DfMethodology.at[8, "FeatureConstruction"] = "FeatureLags= %s" % self.FeatureLag
        DfMethodology.at[
            9, "FeatureConstruction"] = "Automatic creation of time series ownlags= %s" % self.AutomaticTimeSeriesOwnlagConstruct
        if self.AutomaticTimeSeriesOwnlagConstruct == True:
            DfMethodology.at[10, "FeatureConstruction"] = "Minimal Ownlag= %s" % self.MinOwnLag
        DfMethodology.at[11, "FeatureConstruction"] = "Automatic creation of lagged features= %s" % self.AutoFeaturelagCreate
        if self.AutoFeaturelagCreate == True:
            DfMethodology.at[12, "FeatureConstruction"] = "First lag to be considered= %s" % self.MinFeatureLag
            DfMethodology.at[13, "FeatureConstruction"] = "Last lag to be considered= %s" % self.MaxFeatureLag

        DfMethodology.at[1, "FeatureSelection"] = "Manual feature selection = %s" % self.ManFeatureSelect
        if self.ManFeatureSelect == True:
            DfMethodology.at[2, "FeatureSelection"] = "Selected Features= %s" % self.FeatureSelect
        DfMethodology.at[3, "FeatureSelection"] = "Low Variance Filter = %s" % self.LowVarianceFilter
        if self.LowVarianceFilter == True:
            DfMethodology.at[4, "FeatureSelection"] = "Threshold Variance= %s" % self.Threshold_LowVarianceFilter
        DfMethodology.at[5, "FeatureSelection"] = "Independent Component Analysis = %s" % self.ICA
        DfMethodology.at[6, "FeatureSelection"] = "Univariate Filter = %s" % self.UnivariateFilter
        if self.UnivariateFilter == True:
            DfMethodology.at[7, "FeatureSelection"] = "Score function= %s" % self.Score_func
            DfMethodology.at[8, "FeatureSelection"] = "Search mode= %s" % self.SearchMode
            DfMethodology.at[9, "FeatureSelection"] = "Search mode threshold parameter= %s" % self.Param_univariate_filter
        DfMethodology.at[10, "FeatureSelection"] = "Embedded-Recursive Feature Selection = %s" % (
                self.RecursiveFeatureSelection or self.EmbeddedFeatureSelectionThreshold)
        if (self.RecursiveFeatureSelection or self.EmbeddedFeatureSelectionThreshold) == True:
            DfMethodology.at[11, "FeatureSelection"] = "Embedded Estimator = %s" % self.EstimatorEmbedded
            if self.RecursiveFeatureSelection == True:
                DfMethodology.at[12, "FeatureSelection"] = "Number of Features to select= %s" % self.N_feature_to_select_RFE
                if self.N_feature_to_select_RFE == "automatic":
                    DfMethodology.at[13, "FeatureSelection"] = "CrossValidation= %s" % self.CV_DT
                else:
                    DfMethodology.at[14, "FeatureSelection"] = "CrossValidation= None"
            if self.EmbeddedFeatureSelectionThreshold == True:
                DfMethodology.at[15, "FeatureSelection"] = "Feature importance threshold = %s" % self.Threshold_embedded
                DfMethodology.at[16, "FeatureSelection"] = "CrossValidation= None"

        # save this dataframe in an excel
        ExcelFile = os.path.join(self.ResultsFolder, "Settings_%s.xlsx" % (self.NameOfExperiment))
        writer = pd.ExcelWriter(ExcelFile, engine="openpyxl")
        DfMethodology.to_excel(writer, sheet_name="Methodology")
        writer.save()
        writer.close()
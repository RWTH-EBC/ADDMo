import os
import pandas as pd
import numpy as np
import sys
from sklearn_pandas import DataFrameMapper
from BlackBoxes import *
from sklearn.feature_selection import mutual_info_regression, f_regression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from math import log

# Input section needed for DataTuning and ModelTuning
# Set name of the folder where the experiments shall be saved, e.g. the name of the observed data
NameOfData = "TrialInputRWTH"
# Set name of the experiments series
NameOfExperiment = "TrialTunedDataRWTH"

# User Input Section Data Tuning #######################################################################################
if True:  # if True for neat appearance
    # -----------------------Input Section general variables--------------------------
    # The name of the column where the signal in question is safed; count up from 0 in the first column after the index(1st column after index = 0)
    ColumnOfSignal = 0

    # Set Estimator which shall be used in all Wrapper Methods, should be the one used for the final forecast(just copy paste into WrapperParams)
    # Those settings will only be used if a wrapper is used later on, skip if not.

    # Examplary bayesian Hyperparametergrids:
    Hyperparametergrids = {"ANN": hp.choice("number_of_layers",
                                            [
                                                {"1layer": scope.int(hp.qloguniform("1.1", log(1), log(1000), 1))},
                                                {"2layer": [scope.int(hp.qloguniform("1.2", log(1), log(1000), 1)),
                                                            scope.int(hp.qloguniform("2.2", log(1), log(1000), 1))]},
                                                {"3layer": [scope.int(hp.qloguniform("1.3", log(1), log(1000), 1)),
                                                            scope.int(hp.qloguniform("2.3", log(1), log(1000), 1)),
                                                            scope.int(hp.qloguniform("3.3", log(1), log(1000), 1))]}
                                            ]),
                           "SVR": {"C": hp.loguniform("C", log(1e-4), log(1e3)),
                                   "gamma": hp.loguniform("gamma", log(1e-3), log(1e3)),
                                   "epsilon": hp.loguniform("epsilon", log(1e-4), log(1))},
                           "GB": {"n_estimators": scope.int(hp.qloguniform("n_estimators", log(1), log(1e3), 1)),
                                  "max_depth": scope.int(hp.qloguniform("max_depth", log(1), log(100), 1)),
                                  "learning_rate": hp.loguniform("learning_rate", log(1e-2), log(1)),
                                  "loss": hp.choice("loss", ["ls", "lad", "huber", "quantile"])},
                           "Lasso": {"alpha": hp.loguniform("alpha", log(1e-10), log(1e6))},
                           "RF_bay": {"n_estimators": scope.int(hp.qloguniform("n_estimators", log(1), log(100), 1)),
                                      "max_depth": scope.int(hp.qloguniform("max_depth", log(1), log(100), 1))},
                           "RF": None}
    WrapperModels = {"ANN": ann_bayesian_predictor, "GB": gradientboost_bayesian, "Lasso": lasso_bayesian,
                     "SVR": svr_bayesian_predictor, "RF": rf_predictor}
    EstimatorWrapper = WrapperModels[
        "RF"]  # state one blackbox model from "BlackBoxes.py", without parenthesis, e.g. <rf_predictor>
    WrapperParams = [Hyperparametergrids["RF"], None, None,
                     False]  # state the parameters that the model should have . Eg. [None, None, None, False] or [HyperparameterGrid, TimeSeriesSplit(n_splits=3), 30, False]
    # 1st entry = hyperparametergrid
    # 2nd= crossvalidation
    # 3rd= max_eval
    # 4th= recursive (consider turning on if creating ownlags via wrapper; makes rf about 4 times slower, to other models just little influence)
    MinIncrease = 0.005  # minimum difference between two scores(Score-error) to accept a change to the original data, e.g. NoOwnlag + MinIncrease < 1Ownlag in order to add a new ownlag

    # -----------------------Input Section ImportData-------------------------------
    # Define if data should be resampled############################################
    Resample = False
    # If Resample is True the following resolution is required
    Resolution = "60min"  # e.g. "60min" means into buckets of 60minutes, "30s" to seconds
    WayOfResampling = [np.mean, np.mean,
                       np.mean]  # e.g. for a 3 column data set(index not counted):[np.sum, np.mean, np.mean] first column will be summed up all other will be meaned
    # Define way of resampling per column, available: Resample to larger interval: np.sum, np.mean, np.median or a selfdefined aggregation method

    # -----------------------Input Section Preprocessing-------------------------------
    # Initial manual feature selection
    # Manual selection of Features by their Column number
    InitManFeatureSelect = False
    InitFeatures = [1, 2, 3, 8,
                    9]  # e.g.[2, 3] #enter the column number of the features you want to keep(start to count from 0 with first Column after Index, Column of signal needs to be counted, but will be kept in any case)

    # Define how NaN values should be handled
    NaNDealing = "bfill"  # possible "bfill", "ffill", "dropna" or "None" #advised: bfill or ffill

    # Define how data should be scaled and centered, preferenced: by experience: RobustScaling: Reason:(AHU1 Testing)
    StandardScaling = False  # use if your data does not contain outliers
    # or
    RobustScaling = True  # use if your data contains outliers
    #
    NoScaling = False  # use only if your data is already scaled, centered etc. or you do tests!

    # -----------------------Input Section Period Selection--------------------------
    # Time Series Plot
    TimeSeriesPlot = False

    # Manual Period Selection
    ManSelect = False
    StartDate = '2016-06-02 00:00'  # start day of data set         #Datum geändert von 2016-06-02 00:00
    EndDate = '2016-06-16 00:00'  # end day of data set                # 2016-06-16 00:00

    # -----------------------Input Section Feature Construction----------------------
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

    # -----------------------Input Section Feature Selection-------------------------
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
    rf = RandomForestRegressor(max_depth=10e17,  # Max Tiefe
                               random_state=0)  # have to be defined so that they return "feature_importance", more implementation have to be developed
    lasso = linear_model.Lasso(max_iter=10e8)
    EstimatorEmbedded = rf  # e.g. <rf>; set one of the above models to be used

    # Embedded feature selection by setting an importance threshold (not recursive)
    EmbeddedFeatureSelectionThreshold = False
    Threshold_embedded = "median"  # from 0-1 or "median" or "mean"

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
# End of User Input Section#############################################################################################

# User Input Section Model Tuning########################################################################################
if True:  # if True for neat appearance
    # Variables for "ModelTuning.py" (necessary for
    # User Input
    NameOfSubTest = "FinalBayes"
    StartTraining = '2016-08-01 00:00'  # StartTraining = '2017-12-18 00:00'
    EndTraining = '2016-08-07 23:45'  # EndTraining = '2018-01-13 23:45'
    StartTesting = '2016-08-08 00:00'  # StartTesting = '2018-01-14 00:00'
    EndTesting = '2016-08-16 23:45'  # EndTesting = '2018-12-31 23:45'

    # Logging Results computing time over train steps and AIC/BIC over train steps
    logresults = True

    # Set global variables, those variables are for the BlackBox models themselves not for the final bayesian optimization
    GlobalMaxEval_HyParaTuning = 1  # sets the number of evaluations done by the bayesian optimization for each "tuned training" to find the best Hyperparameter, each evaluation is training and testing with cross-validation for one hyperparameter setting
    GlobalCV_MT = 3  # Enter any crossvalidationn method from scikit-learn or any self defined or from elsewhere.
    GlobalRecu = True  # (Boolean) this sets whether the it shall be forecasted recursive or not
    GlobalShuffle = True

    # Settings for regular training without final bayesian optimization (without "automation)
    GlobalIndivModel = "No"  # "week_weekend"; "hourly"; "No"; "byFeature"
    if GlobalIndivModel == "byFeature":
        IndivFeature = "schedule[]"  # copy the name of feature here
        IndivThreshold = 0.5  # state the threshold at which value of that feature the data frame shall be splitted
    OnlyHyPara_Models = ["ANN"]  # array of the blackboxes you want to use
    # Possible entries: ["SVR","NuSVR", "RF", "RF_bay","ANN", "GB", "Lasso", "SVR_grid", "ANN_grid", "RF_grid", "GB_grid", "Lasso_grid"]
    #                  ["ModelSelection"] uses all bayesian models (those without _grid) and returns the best
    OnlyHyPara_KPI = "kpi3"  # choice of KPI that shall be used for optimization
    # possible entries: "acc", "aic", "bic", "kpi1", "kpi2", "kpi3"


    # Final bayesian optimization finds optimal combination of "Individual Model"&"Features"&"Model"
    # Final bayesian optimization parameter
    MaxEval_Bayes = 1
    Model_Bayes = "ModelSelection"
    # possible entries
    # Max_eval_Bayes = int - Number of iterations the bayesian optimization should do for selecting NumberofFeatures, IndivModel, BestModel , the less the less quality but faster
    # Model= "SVR","ANN","GB","RF","Lasso" - choose a model for bayesian optimization (RF is by far the fastest)
    #        "ModelSelection" - bayesian optimization is done with the score of the best model (hence in each iteration all models are calculated)
    #        "Baye" - models are chosen through bayesian optimization as well (consider higher amount of Max_eval_bayes

    # define and set Estimator which shall be used in for the embedded feature selection
    rf = RandomForestRegressor(max_depth=10e17,
                               random_state=0)  # have to be defined so that they return "feature_importance", more implementation have to be developed
    EstimatorEmbedded_FinalBaye = rf  # e.g. <rf>; set one of the above models to be used

    # Parameters for only prediction
    # This is to use after training the models, hence it won´t produce results if there is no trained model safed already(which is done automatically if training one)
    # You define the trained model you want to load through nameofexperiment and NameOfSubTest and the time you want to predict through __StartDateTest and __EndDateTest
    # of course it is necessary that the models have been trained before in the respective nameofdata and nameofexperiment and NameOfSubTest combination
    NameOfOnlyPredict = "Test1"  # use different names if you want to use several only_predicts on the same trained models
    OnlyPredictRecursive = False

    ValidationPeriod = False
    """Set False to have the prediction error on the whole data period (train and test),
    set True to define a test period by yourself(example the whole outhold data).
    With the difference between StartTesting and EndTesting the required prediction horizon is set
    (this period (StartTesting till EndTesting) is also the one being plotted and analysed with the regular measures)
    The defined test period is then split into periods with the length of "horizon", for each horizon the prediction
    error is computed. Of those errors the "mean", "standard deviation" and the "max error" are computed
    (see "Automated Data Driven Modeling of Building Energy Systems via Machine Learning Algorithms" by Martin Rätz for more details)"""
    if ValidationPeriod == True:
        StartTest_onlypredict = '2018-01-14 00:00'
        EndTest_onlypredict = '2018-12-31 23:45'

        # Only necessary for plotting with the style used in the master thesis
        # PAPER_NameOfSignal = "Power"
        # PAPER_UNIT = "kW"
        # PAPER_xUNIT = "h"
# End of User Input Section#############################################################################################

########################################################################################################################
# Some variables that are set automatically by "DataTuning" or "ModelTuning"
NameOfSignal = "Empty"  # is set through both or GUI
RootDir = "Empty"  # by DataTuning
PathToData = "Empty"  # by DataTuning
ResultsFolder = "Empty"  # by DataTuning
PathToPickles = "Empty"  # by DataTuning
ResultsFolderSubTest = "Empty"  # by ModelTuning
OnlyPredictFolder = "Empty"  # by ModelTuning
InputData = "Empty"  # by DataTuning
FixImport = True  # set by GUI used in "DataTuning"
GUI_Filename = "Empty"  # set by GUI
save_datatuning_as_xlsx = "Empty" # set by Datatuning
OnlyHyPara = False  # set by Modeltuning


# Some functions used in many modules:
# get the unit of the meter in question (counted from 0 = first column after index), unit has to be in brackets e.g. [Kwh]
def get_unit_of_meter(Data, ColumnOfMeter):
    UoM = list(Data)[ColumnOfMeter].split("[")[1].split("]")[0]
    return UoM


# get the name of the respective Meter
def nameofmeter(Data, ColumnOfMeter):
    Name = list(Data)[ColumnOfMeter]
    return Name


# split signal from feature for the use in an estimator
def split_signal_and_features(Data):
    X = Data.drop(NameOfSignal, axis=1)
    Y = Data[NameOfSignal]
    return (X, Y)


# merge after an embedded operator modified the datasets
def merge_signal_and_features_embedded(X_Data, Y_Data, support, X_Data_transformed):
    columns = X_Data.columns
    rows = X_Data.index
    labels = [columns[x] for x in support if
              x >= 0]  # get the columns which shall be kept by the transformer(the selected features)
    Features = pd.DataFrame(X_Data_transformed, columns=labels,
                            index=rows)  # creates a dataframe reassigning the names of the features as column header and the index as index
    Signal = pd.DataFrame(Y_Data, columns=[NameOfSignal])  # create dataframe of y
    Data = pd.concat([Signal, Features], axis=1)
    return Data


# regular merge
def merge_signal_and_features(X_Data, Y_Data, X_Data_transformed):
    columns = X_Data.columns
    rows = X_Data.index
    Features = pd.DataFrame(X_Data_transformed, columns=columns,
                            index=rows)  # creates a dataframe reassigning the names of the features as column header and the index as index
    Signal = pd.DataFrame(Y_Data, columns=[NameOfSignal])  # create dataframe of y
    Data = pd.concat([Signal, Features], axis=1)
    return Data


# scaling; used if new "unscaled" features were created throughout Feature Construction
def post_scaler(Data, StandardScaling, RobustScaling):
    # Doing "StandardScaler"
    try:  # works only for dataframes not Series
        if StandardScaling == True:
            mapper = DataFrameMapper([(Data.columns, StandardScaler())])  # create the actually used scaler
            Scaled_Data = mapper.fit_transform(Data.copy())  # train it and scale the data
            Data = pd.DataFrame(Scaled_Data, index=Data.index, columns=Data.columns)
        # Doing "RobustScaler"
        if RobustScaling == True:
            mapper = DataFrameMapper([(Data.columns, RobustScaler())])  # create the actually used scaler
            Scaled_Data = mapper.fit_transform(Data.copy())  # train it and scale the data
            Data = pd.DataFrame(Scaled_Data, index=Data.index, columns=Data.columns)
        return Data
    except:  # for data series
        array = Data.values.reshape(-1, 1)
        if StandardScaling == True:
            mapper = StandardScaler()  # create the actually used scaler
            Scaled_Data = mapper.fit_transform(array)  # train it and scale the data
        # Doing "RobustScaler"
        if RobustScaling == True:
            mapper = RobustScaler()  # create the actually used scaler
            Scaled_Data = mapper.fit_transform(array)  # train it and scale the data
        Scaled_Data = pd.DataFrame(Scaled_Data, index=Data.index)
        return Scaled_Data


def reshape(series):
    '''
    Can reshape pandas series and numpy.array

    :param series:
    :type series: pandas.series or mumpy.ndarray
    :return: two dimensional array with one column (like a series)
    :rtype: ndarray
    '''

    if isinstance(series, pd.Series):
        array = series.values.reshape(-1, 1)
    elif isinstance(series, pd.DataFrame):
        array = series.values.reshape(-1, 1)
    elif isinstance(series, np.ndarray):
        array = series.reshape(-1, 1)
    elif isinstance(series, list):
        array = np.array(series).reshape(-1, 1)
    else:
        print("reshape could not been done, unsupported data type{}".format(type(series)))

    return array


def del_unsupported_os_characters(str):
    str = str.replace("/", "").replace("\\", "").replace(":", "").replace("?", "").replace("*", "").replace("\"",
                                                                                                            "").replace(
        "<", "").replace(">", "").replace("|", "")
    return str


# Documents all settings used in the section Data Tuning
def documentation_DataTuning(timestart, timeend):
    print("Documentation")
    # dump the name of signal in the resultsfolder, so that i can always be pulled whenever you want to come back to that specific "Final Input Data"
    joblib.dump(NameOfSignal, os.path.join(ResultsFolder, "NameOfSignal.save"))

    ######saving the methodology of creating FinalInputData in the ExcelFile "Settings"#####################################
    DfMethodology = pd.DataFrame(index=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                                 columns=["GlobalVariables", "ImportData", "Preprocessing", "PeriodSelection",
                                          "FeatureConstruction", "FeatureSelection"])
    # adding information to the dataframe
    DfMethodology.at[1, "GlobalVariables"] = "NameOfData = %s" % NameOfData
    DfMethodology.at[2, "GlobalVariables"] = "NameOfExperiment = %s" % NameOfExperiment
    DfMethodology.at[3, "GlobalVariables"] = "NameOfSignal = %s" % NameOfSignal
    DfMethodology.at[4, "GlobalVariables"] = "Model used for Wrappers = %s" % EstimatorWrapper.__name__
    DfMethodology.at[5, "GlobalVariables"] = "Parameter used for Wrappers = %s" % WrapperParams
    DfMethodology.at[6, "GlobalVariables"] = "MinIncrease for Wrappers = %s" % MinIncrease
    DfMethodology.at[7, "GlobalVariables"] = "Pipeline took %s seconds" % (timeend - timestart)

    DfMethodology.at[1, "Preprocessing"] = "How to deal NaN´s = %s" % NaNDealing
    DfMethodology.at[2, "Preprocessing"] = "Initial feature select = %s" % InitManFeatureSelect
    if InitManFeatureSelect == True:
        DfMethodology.at[3, "Preprocessing"] = "Features selected = %s" % InitFeatures
    DfMethodology.at[4, "Preprocessing"] = "StandardScaler = %s" % StandardScaling
    DfMethodology.at[5, "Preprocessing"] = "RobustScaler = %s" % RobustScaling
    DfMethodology.at[6, "Preprocessing"] = "NoScaling = %s" % NoScaling
    DfMethodology.at[7, "Preprocessing"] = "Resample = %s" % Resample
    if Resample == True:
        DfMethodology.at[8, "Preprocessing"] = "Resolution = %s" % Resolution
        DfMethodology.at[9, "Preprocessing"] = "WayOfResampling = %s" % WayOfResampling

    DfMethodology.at[1, "PeriodSelection"] = "ManualSelection = %s" % ManSelect
    if ManSelect == True:
        DfMethodology.at[2, "PeriodSelection"] = "%s till %s" % (StartDate, EndDate)
    DfMethodology.at[3, "PeriodSelection"] = "TimeSeriesPlot = %s" % TimeSeriesPlot

    DfMethodology.at[
        1, "FeatureConstruction"] = "Cross, auto, cloud correlation plot= %s" % Cross_auto_cloud_correlation_plotting
    if Cross_auto_cloud_correlation_plotting == True:
        DfMethodology.at[2, "FeatureConstruction"] = "LagsToBePlotted= %s" % LagsToBePlotted
    DfMethodology.at[3, "FeatureConstruction"] = "DifferenceCreate= %s" % DifferenceCreate
    if DifferenceCreate == True:
        Word = "All" if FeaturesDifference == True else FeaturesDifference
        DfMethodology.at[4, "FeatureConstruction"] = "FeaturesToCreateDifference= %s" % Word
    DfMethodology.at[5, "FeatureConstruction"] = "Manual creation of OwnLags= %s" % ManOwnlagCreate
    if ManOwnlagCreate == True:
        DfMethodology.at[6, "FeatureConstruction"] = "OwnLags= %s" % OwnLag
    DfMethodology.at[7, "FeatureConstruction"] = "Manual creation of FeatureLags= %s" % ManFeaturelagCreate
    if ManFeaturelagCreate == True:
        DfMethodology.at[8, "FeatureConstruction"] = "FeatureLags= %s" % FeatureLag
    DfMethodology.at[
        9, "FeatureConstruction"] = "Automatic creation of time series ownlags= %s" % AutomaticTimeSeriesOwnlagConstruct
    if AutomaticTimeSeriesOwnlagConstruct == True:
        DfMethodology.at[10, "FeatureConstruction"] = "Minimal Ownlag= %s" % MinOwnLag
    DfMethodology.at[11, "FeatureConstruction"] = "Automatic creation of lagged features= %s" % AutoFeaturelagCreate
    if AutoFeaturelagCreate == True:
        DfMethodology.at[12, "FeatureConstruction"] = "First lag to be considered= %s" % MinFeatureLag
        DfMethodology.at[13, "FeatureConstruction"] = "Last lag to be considered= %s" % MaxFeatureLag

    DfMethodology.at[1, "FeatureSelection"] = "Manual feature selection = %s" % ManFeatureSelect
    if ManFeatureSelect == True:
        DfMethodology.at[2, "FeatureSelection"] = "Selected Features= %s" % FeatureSelect
    DfMethodology.at[3, "FeatureSelection"] = "Low Variance Filter = %s" % LowVarianceFilter
    if LowVarianceFilter == True:
        DfMethodology.at[4, "FeatureSelection"] = "Threshold Variance= %s" % Threshold_LowVarianceFilter
    DfMethodology.at[5, "FeatureSelection"] = "Independent Component Analysis = %s" % ICA
    DfMethodology.at[6, "FeatureSelection"] = "Univariate Filter = %s" % UnivariateFilter
    if UnivariateFilter == True:
        DfMethodology.at[7, "FeatureSelection"] = "Score function= %s" % Score_func
        DfMethodology.at[8, "FeatureSelection"] = "Search mode= %s" % SearchMode
        DfMethodology.at[9, "FeatureSelection"] = "Search mode threshold parameter= %s" % Param_univariate_filter
    DfMethodology.at[10, "FeatureSelection"] = "Embedded-Recursive Feature Selection = %s" % (
            RecursiveFeatureSelection or EmbeddedFeatureSelectionThreshold)
    if (RecursiveFeatureSelection or EmbeddedFeatureSelectionThreshold) == True:
        DfMethodology.at[11, "FeatureSelection"] = "Embedded Estimator = %s" % EstimatorEmbedded
        if RecursiveFeatureSelection == True:
            DfMethodology.at[12, "FeatureSelection"] = "Number of Features to select= %s" % N_feature_to_select_RFE
            if N_feature_to_select_RFE == "automatic":
                DfMethodology.at[13, "FeatureSelection"] = "CrossValidation= %s" % CV_DT
            else:
                DfMethodology.at[14, "FeatureSelection"] = "CrossValidation= None"
        if EmbeddedFeatureSelectionThreshold == True:
            DfMethodology.at[15, "FeatureSelection"] = "Feature importance threshold = %s" % Threshold_embedded
            DfMethodology.at[16, "FeatureSelection"] = "CrossValidation= None"

    # save this dataframe in an excel
    ExcelFile = os.path.join(ResultsFolder, "Settings_%s.xlsx" % (NameOfExperiment))
    writer = pd.ExcelWriter(ExcelFile, engine="openpyxl")
    DfMethodology.to_excel(writer, sheet_name="Methodology")
    writer.save()
    writer.close()
    ########################################################################################################################

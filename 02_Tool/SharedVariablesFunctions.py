import os
import shutil
import pandas as pd
import numpy as np
import sys
from sklearn_pandas import DataFrameMapper
from PredictorDefinitions import *
from Functions.ErrorMetrics import *
from Functions.PlotFcn import *
from Functions.ErrorMetrics import *
from Functions.PlotFcn import *
from sklearn.feature_selection import mutual_info_regression, f_regression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from math import log

from DataTuningSetup import DataTuningSetup as DTS
from ModelTuningSetup import ModelTuningSetup as MTS

GUI_Filename = "Empty"

Hyperparametergrids = {
    "ANN": hp.choice(
        "number_of_layers",
        [
            {"1layer": scope.int(hp.qloguniform("1.1", log(1), log(1000), 1))},
            {
                "2layer": [
                    scope.int(hp.qloguniform("1.2", log(1), log(1000), 1)),
                    scope.int(hp.qloguniform("2.2", log(1), log(1000), 1)),
                ]
            },
            {
                "3layer": [
                    scope.int(hp.qloguniform("1.3", log(1), log(1000), 1)),
                    scope.int(hp.qloguniform("2.3", log(1), log(1000), 1)),
                    scope.int(hp.qloguniform("3.3", log(1), log(1000), 1)),
                ]
            },
        ],
    ),
    "SVR": {
        "C": hp.loguniform("C", log(1e-4), log(1e4)),
        "gamma": hp.loguniform("gamma", log(1e-3), log(1e4)),
        "epsilon": hp.loguniform("epsilon", log(1e-4), log(1)),
    },
    "GB": {
        "n_estimators": scope.int(hp.qloguniform("n_estimators", log(1), log(1e3), 1)),
        "max_depth": scope.int(hp.qloguniform("max_depth", log(1), log(100), 1)),
        "learning_rate": hp.loguniform("learning_rate", log(1e-2), log(1)),
        "loss": hp.choice("loss", ["ls", "lad", "huber", "quantile"]),
    },
    "Lasso": {"alpha": hp.loguniform("alpha", log(1e-10), log(1e6))},
    "RF": None,
}
WrapperModels = {
    "ANN": ann_bayesian_predictor,
    "GB": gradientboost_bayesian,
    "Lasso": lasso_bayesian,
    "SVR": svr_bayesian_predictor,
    "RF": rf_predictor,
}

rf = RandomForestRegressor(
    max_depth=10e17, random_state=0
)  # have to be defined so that they return "feature_importance", more implementation have to be developed


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
def split_signal_and_features(NameOfSignal, Data):
    X = Data.drop(NameOfSignal, axis=1)
    Y = Data[NameOfSignal]
    return (X, Y)


# merge after an embedded operator modified the datasets
def merge_signal_and_features_embedded(
    NameOfSignal, X_Data, Y_Data, support, X_Data_transformed
):
    columns = X_Data.columns
    rows = X_Data.index
    labels = [
        columns[x] for x in support if x >= 0
    ]  # get the columns which shall be kept by the transformer(the selected features)
    Features = pd.DataFrame(
        X_Data_transformed, columns=labels, index=rows
    )  # creates a dataframe reassigning the names of the features as column header and the index as index
    Signal = pd.DataFrame(Y_Data, columns=[NameOfSignal])  # create dataframe of y
    Data = pd.concat([Signal, Features], axis=1)
    return Data


# regular merge
def merge_signal_and_features(NameOfSignal, X_Data, Y_Data, X_Data_transformed):
    columns = X_Data.columns
    rows = X_Data.index
    Features = pd.DataFrame(
        X_Data_transformed, columns=columns, index=rows
    )  # creates a dataframe reassigning the names of the features as column header and the index as index
    Signal = pd.DataFrame(Y_Data, columns=[NameOfSignal])  # create dataframe of y
    Data = pd.concat([Signal, Features], axis=1)
    return Data


# scaling; used if new "unscaled" features were created throughout Feature Construction
def post_scaler(Data, StandardScaling, RobustScaling):
    # Doing "StandardScaler"
    try:  # works only for dataframes not Series
        if StandardScaling == True:
            mapper = DataFrameMapper(
                [(Data.columns, StandardScaler())]
            )  # create the actually used scaler
            Scaled_Data = mapper.fit_transform(
                Data.copy()
            )  # train it and scale the data
            Data = pd.DataFrame(Scaled_Data, index=Data.index, columns=Data.columns)
        # Doing "RobustScaler"
        if RobustScaling == True:
            mapper = DataFrameMapper(
                [(Data.columns, RobustScaler())]
            )  # create the actually used scaler
            Scaled_Data = mapper.fit_transform(
                Data.copy()
            )  # train it and scale the data
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
    """
    Can reshape pandas series and numpy.array

    :param series:
    :type series: pandas.series or mumpy.ndarray
    :return: two dimensional array with one column (like a series)
    :rtype: ndarray
    """

    if isinstance(series, pd.Series):
        array = series.values.reshape(-1, 1)
    elif isinstance(series, pd.DataFrame):
        array = series.values.reshape(-1, 1)
    elif isinstance(series, np.ndarray):
        array = series.reshape(-1, 1)
    elif isinstance(series, list):
        array = np.array(series).reshape(-1, 1)
    else:
        print(
            "reshape could not been done, unsupported data type{}".format(type(series))
        )

    return array


def del_unsupported_os_characters(str):
    str = (
        str.replace("/", "")
        .replace("\\", "")
        .replace(":", "")
        .replace("?", "")
        .replace("*", "")
        .replace('"', "")
        .replace("<", "")
        .replace(">", "")
        .replace("|", "")
    )
    return str


def delete_and_create_folder(path):
    """Make sure not to accidentally overwrite data."""

    if os.path.isdir(path) == True:
        answer = input(
            f"You are about to overwrite already existing data. Do you want to delete the data in: \n'{path}'?:\n"
        )
        if answer == "y":
            shutil.rmtree(path)
            print("Folder deleted. Script continued.")
        else:
            sys.exit("Code stopped by user. Enter y for yes\n")
    os.makedirs(path)


def getscores(Y_test, Y_Predicted):
    # evaluate results
    (R2, STD, RMSE, MAPE, MAE) = evaluation(Y_test, Y_Predicted)

    # return Scores for modelselection
    return R2, STD, RMSE, MAPE, MAE


def apply_scaler(MT_Setup_Object, Y_test, Y_Predicted, Indexer):
    if os.path.isfile(
        os.path.join(MT_Setup_Object.ResultsFolder, "ScalerTracker.save")
    ):  # if scaler was used
        ScaleTracker_Signal = joblib.load(
            os.path.join(MT_Setup_Object.ResultsFolder, "ScalerTracker.save")
        )  # load used scaler
        # Scale Results back to normal; maybe inside the Blackboxes
        Y_Predicted = ScaleTracker_Signal.inverse_transform(reshape(Y_Predicted))
        Y_test = ScaleTracker_Signal.inverse_transform(reshape(Y_test))
        # convert arrays to data frames(Series) for further use
        Y_test = pd.DataFrame(index=Indexer, data=Y_test, columns=["Measure"])
        Y_test = Y_test["Measure"]

    # convert arrays to data frames(Series) for further use
    Y_Predicted = pd.DataFrame(index=Indexer, data=Y_Predicted, columns=["Prediction"])
    Y_Predicted = Y_Predicted["Prediction"]

    return Y_test, Y_Predicted


class setup_object_initializer:
    def __init__(self, Object):
        # define path to data source files '.xls' & '.pickle'
        self.RootDir = os.path.dirname(os.path.realpath(__file__))
        self.PathToData = os.path.join(self.RootDir, "Data")
        self.Object = Object

    # -------------------------------DTS initialisations-----------------------------------------------------------
    def dts(self):
        DT_Setup_Object = self.Object
        # Set Folder for Results
        ResultsFolder = os.path.join(
            self.RootDir,
            "Results",
            DT_Setup_Object.NameOfData,
            DT_Setup_Object.NameOfExperiment,
        )
        PathToPickles = os.path.join(ResultsFolder, "Pickles")

        # Set the found Variables in "SharedVariables"
        DT_Setup_Object.RootDir = self.RootDir
        DT_Setup_Object.PathToData = self.PathToData
        DT_Setup_Object.ResultsFolder = ResultsFolder
        DT_Setup_Object.PathToPickles = PathToPickles

        # Runtime variable initializations
        DT_Setup_Object.FixImport = False
        DT_Setup_Object.InputData = "Empty"

        return DT_Setup_Object

    # -------------------------------MTS initialisations-----------------------------------------------------------
    def mts(self):
        MT_Setup_Object = self.Object
        # Set Folder for Results
        ResultsFolder = os.path.join(
            self.RootDir,
            "Results",
            MT_Setup_Object.NameOfData,
            MT_Setup_Object.NameOfExperiment,
        )
        PathToPickles = os.path.join(ResultsFolder, "Pickles")

        # Set the found Variables in "SharedVariables"
        MT_Setup_Object.RootDir = self.RootDir
        MT_Setup_Object.PathToData = self.PathToData
        MT_Setup_Object.ResultsFolder = ResultsFolder
        MT_Setup_Object.PathToPickles = PathToPickles

        # Runtime variable initializations
        MT_Setup_Object.ResultsFolderSubTest = "Empty"

        return MT_Setup_Object

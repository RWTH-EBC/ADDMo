from math import log
from sklearn.metrics import r2_score
from hyperopt.pyll import scope
from sklearn.externals import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils import shuffle
import os
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import pandas as pd
import numpy as np
from pandas.io.excel import ExcelWriter
# from GlobalVariables import *
from openpyxl import load_workbook
import sys
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import FastICA
from sklearn.feature_selection import GenericUnivariateSelect
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split

from BlackBoxes import *
from PredictorDefinitions import *
from Functions.ErrorMetrics import *
from Functions.PlotFcn import *

import SharedVariablesFunctions as SVF
from ModelTuningSetup import ModelTuningSetup as MTS
from ModelTuningRuntimeResults import ModelTuningRuntimeResults as MTRR

print("Model Tuning Started")

# --------------------------------------------- Model Tuning Section --------------------------------------------------

# Function used by Hyperopt, OnlyPredict and AFB (hence the parameter MT_Setup_Object is generically used for
# MT_Setup_Object_X type objects, where X refers to Hyperopt/PO/AFB based on the function call)

def pre_handling(MT_Setup_object, OnlyPredict):
    # define path to data source files '.xls' & '.pickle'
    RootDir = os.path.dirname(os.path.realpath(__file__)) #Todo: could all that folder "pathing" be done within the setup class (as function eg. in the init?)?
    PathToData = os.path.join(RootDir, 'Data')

    # Set Folder for Results
    ResultsFolder = os.path.join(RootDir, "Results", MT_Setup_object.NameOfData, MT_Setup_object.NameOfExperiment)
    PathToPickles = os.path.join(ResultsFolder, "Pickles")

    MT_Setup_object.RootDir = RootDir
    MT_Setup_object.PathToData = PathToData
    MT_Setup_object.ResultsFolder = ResultsFolder
    MT_Setup_object.PathToPickles = PathToPickles

    ResultsFolderSubTest = os.path.join(MT_Setup_object.ResultsFolder, 'Predictions', MT_Setup_object.NameOfSubTest)
    MT_Setup_object.ResultsFolderSubTest = ResultsFolderSubTest

    # check if experiment folder is present
    if os.path.isdir(MT_Setup_object.ResultsFolder) == False:
        sys.exit("Set a valid experiment folder via NameOfData and NameOfExperiment.")

    # check if test results are saved in the right folder:
    if OnlyPredict != True:
        SVF.delete_and_create_folder(MT_Setup_object.ResultsFolderSubTest)

    # Take Tuned data, build Train and Test Sets, and split them into signal and features
    NameOfSignal = joblib.load(os.path.join(MT_Setup_object.ResultsFolder, "NameOfSignal.save"))
    MT_Setup_object.NameOfSignal = NameOfSignal  # Todo: check whether NameOfSignal fits with GUI (maybe one wants to define it by himself)

    # Take FinalInputData, build Train and Test Sets, and split them into signal and features
    if OnlyPredict == True:
        ImportBaye = os.path.isfile(os.path.join(MT_Setup_object.ResultsFolderSubTest, "BestData_%s.xlsx" % (
            MT_Setup_object.NameOfSubTest)))  # is True if FinalBayes was used, this implies that we want to load the data that was produced by finalbayes
    else:
        ImportBaye = False  # if onlypredict isn´t used we (up to now) don´t want to load from finalbayes
    if ImportBaye == False:
        Data = pd.read_pickle(os.path.join(MT_Setup_object.PathToPickles,
                                           "ThePickle_from_FeatureSelection" + '.pickle'))  # import from data tuning
    if ImportBaye == True:
        Data = pd.read_pickle(os.path.join(MT_Setup_object.PathToPickles,
                                           "ThePickle_from_%s" % MT_Setup_object.NameOfSubTest + '.pickle'))  # import the data set produced by "final bayesian optimization"

    (Data_Train, Data_Test) = manual_train_test_period_select(Data=Data, StartDateTrain=MT_Setup_object.StartTraining,
                                                              EndDateTrain=MT_Setup_object.EndTraining,
                                                              StartDateTest=MT_Setup_object.StartTesting,
                                                              EndDateTest=MT_Setup_object.EndTesting)

    # shuffles data randomly if wished
    if MT_Setup_object.GlobalShuffle == True:
        Data_Train = shuffle(Data_Train)
        # Data_Test = shuffle(Data_Test) #not necessary since experiments showed that the order of test samples does not affect the >prediction<

    (_X_train, _Y_train) = SVF.split_signal_and_features(MT_Setup_object.NameOfSignal, Data_Train)
    (_X_test, _Y_test) = SVF.split_signal_and_features(MT_Setup_object.NameOfSignal, Data_Test)
    Indexer = _X_test.index  # for tracking the orignal index(timestamps) of the test data

    return _X_train, _Y_train, _X_test, _Y_test, Indexer, Data


def manual_train_test_period_select(Data, StartDateTrain, EndDateTrain, StartDateTest, EndDateTest):
    Data_TrainTest = Data[StartDateTrain:EndDateTrain]  # is used to train the model and evaluate the hyperparameter
    Data_Test = Data[StartDateTest:EndDateTest]  # is used to perform a "forecast" with the trained Model
    return (Data_TrainTest, Data_Test)


def main_OnlyHyParaOpti(MT_Setup_Object):
    print("Start training and testing with only optimizing the hyperparameters: %s/%s/%s" % (
        MT_Setup_Object.NameOfData, MT_Setup_Object.NameOfExperiment, MT_Setup_Object.NameOfSubTest))
    _X_train, _Y_train, _X_test, _Y_test, Indexer, Data = pre_handling(MT_Setup_Object, OnlyPredict=False)

    MT_RR_object = MTRR()

    train_predict_selected_models(MT_Setup_Object, MT_RR_object, _X_train, _Y_train, _X_test, _Y_test, Indexer,
                                  MT_Setup_Object.GlobalIndivModel, True)

    print("Finish training and testing with only optimizing the hyperparameters : %s/%s/%s" % (
        MT_Setup_Object.NameOfData, MT_Setup_Object.NameOfExperiment, MT_Setup_Object.NameOfSubTest))
    print("________________________________________________________________________\n")
    print("________________________________________________________________________\n")
    MT_RR_object.store_results(MT_Setup_Object)


# # def modelselection(MT_Setup_Object, MT_RR_object, _X_train, _Y_train, _X_test, _Y_test, Indexer="IndexerError",
# #                    IndividualModel="Error",
# #                    Documentation=False):
# #     # Trains and tests all (bayesian) models and returns the best of them, also saves it in an txtfile.
# #     Score_RF = BB3.train_predict(MT_Setup_Object, MT_RR_object.RF_Summary, _X_train, _Y_train, _X_test, _Y_test,
# #                                  Indexer, IndividualModel,
# #                                  Documentation)
# #     Score_ANN = BB5.train_predict(MT_Setup_Object, MT_RR_object.ANN_Summary, _X_train, _Y_train, _X_test, _Y_test,
# #                                   Indexer, IndividualModel,
# #                                   Documentation)
# #     Score_GB = BB7.train_predict(MT_Setup_Object, MT_RR_object.GB_Summary, _X_train, _Y_train, _X_test, _Y_test,
# #                                  Indexer, IndividualModel,
# #                                  Documentation)
# #     Score_Lasso = BB9.train_predict(MT_Setup_Object, MT_RR_object.Lasso_Summary, _X_train, _Y_train, _X_test, _Y_test,
# #                                     Indexer, IndividualModel,
# #                                     Documentation)
# #     Score_SVR = BB2.train_predict(MT_Setup_Object, MT_RR_object.SVR_Summary, _X_train, _Y_train, _X_test, _Y_test,
# #                                   Indexer, IndividualModel,
# #                                   Documentation)
#
#     Score_list = [0, 1, 2, 3, 4]
#     Score_list[0] = Score_SVR
#     Score_list[1] = Score_RF
#     Score_list[2] = Score_ANN
#     Score_list[3] = Score_GB
#     Score_list[4] = Score_Lasso
#
#     print(Score_list)
#     # Todo: if Scoring function Score max; if Scoring function some error: min
#     BestScore = max(Score_list)
#
#     if Score_list[0] == BestScore:
#         __BestModel = "SVR"
#     if Score_list[1] == BestScore:
#         __BestModel = "RF"
#     if Score_list[2] == BestScore:
#         __BestModel = "ANN"
#     if Score_list[3] == BestScore:
#         __BestModel = "GB"
#     if Score_list[4] == BestScore:
#         __BestModel = "Lasso"
#
#     # state best model in txt file
#     f = open(os.path.join(MT_Setup_Object.ResultsFolderSubTest, "BestModel.txt"), "w+")
#     f.write("The best model is %s with an accuracy of %s" % (__BestModel, BestScore))
#     f.close()
#     return BestScore


if __name__ == '__main__':
    # Todo: The following is done in ModelTuning and DataTuning, isn´t it better once in SVF?

    MT_Setup_Object = MTS()
    # define path to data source files '.xls' & '.pickle'
    RootDir = os.path.dirname(os.path.realpath(__file__))
    PathToData = os.path.join(RootDir, 'Data')

    # Set Folder for Results
    ResultsFolder = os.path.join(RootDir, "Results", MT_Setup_Object.NameOfData, MT_Setup_Object.NameOfExperiment)
    PathToPickles = os.path.join(ResultsFolder, "Pickles")

    # Set the found Variables in "SharedVariables"
    MT_Setup_Object.RootDir = RootDir
    MT_Setup_Object.PathToData = PathToData
    MT_Setup_Object.ResultsFolder = ResultsFolder
    MT_Setup_Object.PathToPickles = PathToPickles

    # Define which part shall be computed (parameters are set in SharedVariables)
    # main_FinalBayes()
    main_OnlyHyParaOpti(MT_Setup_Object)
    # main_OnlyPredict()

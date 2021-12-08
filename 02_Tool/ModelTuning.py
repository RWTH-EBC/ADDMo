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
from BlackBoxes_mkm import *
from Functions.ErrorMetrics import *
from Functions.PlotFcn import *

import SharedVariables as SV
from ModelTuningSetup import ModelTuningSetup as MTS
from ModelTuningRuntimeResults import ModelTuningRuntimeResults as MTRR

print("Start")


# -------------------------------------------------------------------------------------------------------------------

# saves the BestModels in a folder "BestModels", also capable of saving individual models
def model_saver(Result_dic, ResultsFolderSubTest, NameOfPredictor, IndividualModel):
    if os.path.isdir(os.path.join(ResultsFolderSubTest, "BestModels")) == True:
        pass
    else:
        os.makedirs(os.path.join(ResultsFolderSubTest, "BestModels"))

    if IndividualModel == "week_weekend":
        joblib.dump(Result_dic["Best_trained_model"]["weekday"], os.path.join(ResultsFolderSubTest, "BestModels",
                                                                              "weekday_%s.save" % (
                                                                                  NameOfPredictor)))  # dump the best trained model in a file to reuse it for different predictions
        joblib.dump(Result_dic["Best_trained_model"]["weekend"], os.path.join(ResultsFolderSubTest, "BestModels",
                                                                              "weekend_%s.save" % (
                                                                                  NameOfPredictor)))  # dump the best trained model in a file to reuse it for different predictions
    elif IndividualModel == "hourly":
        joblib.dump(Result_dic["Best_trained_model"][0],
                    os.path.join(ResultsFolderSubTest, "BestModels", "0_%s.save" % (NameOfPredictor)))
        joblib.dump(Result_dic["Best_trained_model"][1],
                    os.path.join(ResultsFolderSubTest, "BestModels", "1_%s.save" % (NameOfPredictor)))
        joblib.dump(Result_dic["Best_trained_model"][2],
                    os.path.join(ResultsFolderSubTest, "BestModels", "2_%s.save" % (NameOfPredictor)))
        joblib.dump(Result_dic["Best_trained_model"][3],
                    os.path.join(ResultsFolderSubTest, "BestModels", "3_%s.save" % (NameOfPredictor)))
        joblib.dump(Result_dic["Best_trained_model"][4],
                    os.path.join(ResultsFolderSubTest, "BestModels", "4_%s.save" % (NameOfPredictor)))
        joblib.dump(Result_dic["Best_trained_model"][5],
                    os.path.join(ResultsFolderSubTest, "BestModels", "5_%s.save" % (NameOfPredictor)))
        joblib.dump(Result_dic["Best_trained_model"][6],
                    os.path.join(ResultsFolderSubTest, "BestModels", "6_%s.save" % (NameOfPredictor)))
        joblib.dump(Result_dic["Best_trained_model"][7],
                    os.path.join(ResultsFolderSubTest, "BestModels", "7_%s.save" % (NameOfPredictor)))
        joblib.dump(Result_dic["Best_trained_model"][8],
                    os.path.join(ResultsFolderSubTest, "BestModels", "8_%s.save" % (NameOfPredictor)))
        joblib.dump(Result_dic["Best_trained_model"][9],
                    os.path.join(ResultsFolderSubTest, "BestModels", "9_%s.save" % (NameOfPredictor)))
        joblib.dump(Result_dic["Best_trained_model"][10],
                    os.path.join(ResultsFolderSubTest, "BestModels", "10_%s.save" % (NameOfPredictor)))
        joblib.dump(Result_dic["Best_trained_model"][11],
                    os.path.join(ResultsFolderSubTest, "BestModels", "11_%s.save" % (NameOfPredictor)))
        joblib.dump(Result_dic["Best_trained_model"][12],
                    os.path.join(ResultsFolderSubTest, "BestModels", "12_%s.save" % (NameOfPredictor)))
        joblib.dump(Result_dic["Best_trained_model"][13],
                    os.path.join(ResultsFolderSubTest, "BestModels", "13_%s.save" % (NameOfPredictor)))
        joblib.dump(Result_dic["Best_trained_model"][14],
                    os.path.join(ResultsFolderSubTest, "BestModels", "14_%s.save" % (NameOfPredictor)))
        joblib.dump(Result_dic["Best_trained_model"][15],
                    os.path.join(ResultsFolderSubTest, "BestModels", "15_%s.save" % (NameOfPredictor)))
        joblib.dump(Result_dic["Best_trained_model"][16],
                    os.path.join(ResultsFolderSubTest, "BestModels", "16_%s.save" % (NameOfPredictor)))
        joblib.dump(Result_dic["Best_trained_model"][17],
                    os.path.join(ResultsFolderSubTest, "BestModels", "17_%s.save" % (NameOfPredictor)))
        joblib.dump(Result_dic["Best_trained_model"][18],
                    os.path.join(ResultsFolderSubTest, "BestModels", "18_%s.save" % (NameOfPredictor)))
        joblib.dump(Result_dic["Best_trained_model"][19],
                    os.path.join(ResultsFolderSubTest, "BestModels", "19_%s.save" % (NameOfPredictor)))
        joblib.dump(Result_dic["Best_trained_model"][20],
                    os.path.join(ResultsFolderSubTest, "BestModels", "20_%s.save" % (NameOfPredictor)))
        joblib.dump(Result_dic["Best_trained_model"][21],
                    os.path.join(ResultsFolderSubTest, "BestModels", "21_%s.save" % (NameOfPredictor)))
        joblib.dump(Result_dic["Best_trained_model"][22],
                    os.path.join(ResultsFolderSubTest, "BestModels", "22_%s.save" % (NameOfPredictor)))
        joblib.dump(Result_dic["Best_trained_model"][23],
                    os.path.join(ResultsFolderSubTest, "BestModels", "23_%s.save" % (NameOfPredictor)))
    elif IndividualModel == "byFeature":
        joblib.dump(Result_dic["Best_trained_model"]["above"],
                    os.path.join(ResultsFolderSubTest, "BestModels", "above_%s.save" % (NameOfPredictor)))
        joblib.dump(Result_dic["Best_trained_model"]["below"],
                    os.path.join(ResultsFolderSubTest, "BestModels", "below_%s.save" % (NameOfPredictor)))
    else:
        joblib.dump(Result_dic["Best_trained_model"],
                    os.path.join(ResultsFolderSubTest, "BestModels", "%s.save" % (NameOfPredictor)))


def getscore(MT_Setup_Object_PO, Y_Predicted, Y_test, Indexer):
    if os.path.isfile(os.path.join(MT_Setup_Object_PO.ResultsFolder, "ScalerTracker.save")):  # if scaler was used
        ScaleTracker_Signal = joblib.load(
            os.path.join(MT_Setup_Object_PO.ResultsFolder, "ScalerTracker.save"))  # load used scaler
        # Scale Results back to normal; maybe inside the Blackboxes
        Y_Predicted = ScaleTracker_Signal.inverse_transform(SV.reshape(Y_Predicted))
        Y_test = ScaleTracker_Signal.inverse_transform(SV.reshape(Y_test))
        # convert arrays to data frames(Series) for further use
        Y_test = pd.DataFrame(index=Indexer, data=Y_test, columns=["Measure"])
        Y_test = Y_test["Measure"]

    # convert arrays to data frames(Series) for further use
    Y_Predicted = pd.DataFrame(index=Indexer, data=Y_Predicted, columns=["Prediction"])
    Y_Predicted = Y_Predicted["Prediction"]

    # evaluate results
    R2 = r2_score(Y_test, Y_Predicted)
    # return Score for modelselection
    return R2

#-----------------------------------------------------------------------------------------------------------------------

# --------------------------------------------- Model Tuning Section --------------------------------------------------

# Function used by Hyperopt, OnlyPredict and AFB (hence MT_Setup_Object is in-fact MT_Setup_Object_X, where X refers
# to Hyperopt/PO/AFB)

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
        SV.delete_and_create_folder(MT_Setup_object.ResultsFolderSubTest)

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

    (_X_train, _Y_train) = SV.split_signal_and_features(MT_Setup_object.NameOfSignal, Data_Train)
    (_X_test, _Y_test) = SV.split_signal_and_features(MT_Setup_object.NameOfSignal, Data_Test)
    Indexer = _X_test.index  # for tracking the orignal index(timestamps) of the test data

    return _X_train, _Y_train, _X_test, _Y_test, Indexer, Data


def manual_train_test_period_select(Data, StartDateTrain, EndDateTrain, StartDateTest, EndDateTest):
    Data_TrainTest = Data[StartDateTrain:EndDateTrain]  # is used to train the model and evaluate the hyperparameter
    Data_Test = Data[StartDateTest:EndDateTest]  # is used to perform a "forecast" with the trained Model
    return (Data_TrainTest, Data_Test)


def main_OnlyHyParaOpti(MT_Setup_Object):
    print("Start training and testing with only optimizing the hyperparameters: %s/%s/%s" % (
        MT_Setup_Object.NameOfData, MT_Setup_Object.NameOfExperiment, MT_Setup_Object.NameOfSubTest))
    _X_train, _Y_train, _X_test, _Y_test, Indexer, Data = pre_handling(MT_Setup_Object, False)

    MT_RR_object = MTRR()

    train_predict_selected_models(MT_Setup_Object, MT_RR_object, _X_train, _Y_train, _X_test, _Y_test, Indexer,
                                  MT_Setup_Object.GlobalIndivModel, True)

    #for Model in MT_Setup_Object.OnlyHyPara_Models:
     #   all_models(MT_Setup_Object, MT_RR_object, Model, _X_train, _Y_train, _X_test, _Y_test, Indexer,
      #             MT_Setup_Object.GlobalIndivModel, True)

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
    # Todo: The following is done in ModelTuning and DataTuning, isn´t it better once in SV?

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

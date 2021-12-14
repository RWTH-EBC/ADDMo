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
import ModelTuning as MT
from ModelTuningRuntimeResults import ModelTuningRuntimeResults as MTRR


def only_predict(MT_Setup_object_PO, RR_Model_Summary, NameOfPredictor, _X_test, _Y_test, Indexer, Data):
    timestart = time.time()
    Predicted, IndividualModel = predict(MT_Setup_object_PO, NameOfPredictor, _X_test)
    if type(Predicted) == bool:
        print(
            "There is no trained model of %s to do OnlyPredict, if needed set OnlyPredict=False and train a model first."
            % NameOfPredictor)  # stop function if specific BestModel is not present
        return
    timeend = time.time()
    ComputationTime = (timeend - timestart)

    MT.visualization_documentation(MT_Setup_object_PO, RR_Model_Summary, NameOfPredictor, Predicted,
                                   _Y_test, Indexer, None, ComputationTime, None, MT_Setup_object_PO.OnlyPredictFolder,
                                   None, None, None, None, MT_Setup_object_PO.OnlyPredictRecursive, IndividualModel,
                                   None)

    def documenation_iterative_evaluation(mean_score, SD_score, errorlist, errormetric):
        errorlist = np.around(errorlist, 3)
        # save results of iterative evaluation in the summary file
        ExcelFile = os.path.join(MT_Setup_object_PO.OnlyPredictFolder,
                                 "Summary_%s_%s.xlsx" % (NameOfPredictor, MT_Setup_object_PO.NameOfSubTest))
        Excel = pd.read_excel(ExcelFile)
        book = load_workbook(ExcelFile)
        writer = pd.ExcelWriter(ExcelFile, engine="openpyxl")
        writer.book = book
        writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
        # create dataframe containing the information
        ErrorDF = pd.DataFrame(index=[0])
        ErrorDF['________'] = "_________________________________"
        if MT_Setup_object_PO.ValidationPeriod == True:
            ErrorDF['Test Data'] = "Interpretation of error measures of the data from %s till %s, per error metric" % (
                MT_Setup_object_PO.StartTest_onlypredict, MT_Setup_object_PO.EndTest_onlypredict)
        else:
            ErrorDF['Test Data'] = "Interpretation of error measures regarding the whole data set per error metric"
        ErrorDF['Used error metric'] = str(errormetric)
        ErrorDF['Horizon length'] = horizon
        ErrorDF['Mean score'] = "%.3f" % mean_score
        ErrorDF['Standard deviation of errors'] = SD_score
        ErrorDF['Max score'] = str(max(errorlist))
        ErrorDF['Min score'] = str(min(errorlist))
        ErrorDF['Number of tested folds'] = len(errorlist)
        ErrorDF = ErrorDF.T

        ErrorListDF = pd.DataFrame(index=range(len(errorlist)))
        ErrorListDF['List of errors'] = errorlist
        ErrorListDF = ErrorListDF.T

        Excel = pd.concat([Excel, ErrorDF, ErrorListDF])

        Excel.to_excel(writer, sheet_name="Sheet1")
        writer.save()
        writer.close()

    horizon = len(_X_test)  # gets the length of the horizon by the stated period to predict
    if MT_Setup_object_PO.ValidationPeriod == True:  # define the data that shall be used to do the mean errors
        MeanErrorData = Data[MT_Setup_object_PO.StartTest_onlypredict:MT_Setup_object_PO.EndTest_onlypredict]
    else:
        MeanErrorData = Data
    fold_list = iterative_evaluation(MT_Setup_object_PO, TestData=MeanErrorData, Model=predict, horizon=horizon,
                                     NameOfPredictor=NameOfPredictor)

    mean_score, SD_score, errorlist, errormetric = mean_scoring(fold_list=fold_list, errormetric=r2_score)
    documenation_iterative_evaluation(mean_score, SD_score, errorlist, "R2")

    mean_score, SD_score, errorlist, errormetric = mean_scoring(fold_list=fold_list, errormetric=mean_absolute_error)
    documenation_iterative_evaluation(mean_score, SD_score, errorlist, "MAE")

    mean_score, SD_score, errorlist, errormetric = mean_scoring(fold_list=fold_list,
                                                                errormetric=mean_absolute_percentage_error)
    documenation_iterative_evaluation(mean_score, SD_score, errorlist, "MAPE")


def predict(MT_Setup_object_PO, NameOfPredictor, _X_test):
    'Loads trained models from previous trainings and does a prediction for the respective period of _X_test.'
    'Individual models are regarded.'
    if os.path.isfile(os.path.join(MT_Setup_object_PO.ResultsFolderSubTest, "BestModels",
                                   "%s.save" % (NameOfPredictor))):  # to find out which indivmodel was used
        Predictor = joblib.load(os.path.join(MT_Setup_object_PO.ResultsFolderSubTest, "BestModels", "%s.save" % (
            NameOfPredictor)))  # load the best and trained model from previous tuning and training
        if MT_Setup_object_PO.OnlyPredictRecursive == False:
            Predicted = Predictor.predict(_X_test)
        elif MT_Setup_object_PO.OnlyPredictRecursive == True:
            Features_test_i = recursive(_X_test, Predictor)
            Predicted = Predictor.predict(Features_test_i)
        IndividualModel = "None"
    elif os.path.isfile(os.path.join(MT_Setup_object_PO.ResultsFolderSubTest, "BestModels",
                                     "23_%s.save" % (NameOfPredictor))):  # for hourly models
        indiv_predictor = indiv_model_onlypredict(indiv_splitter_instance=indiv_splitter(hourly_splitter),
                                                  Features_test=_X_test,
                                                  ResultsFolderSubTest=MT_Setup_object_PO.ResultsFolderSubTest,
                                                  NameOfPredictor=NameOfPredictor,
                                                  Recursive=MT_Setup_object_PO.OnlyPredictRecursive)
        Predicted = indiv_predictor.main()
        IndividualModel = "hourly"
        # Predicted = individual_model_per_hour_onlypredict(_X_test, ResultsFolderSubTest, NameOfPredictor, OnlyPredictRecursive)
    elif os.path.isfile(os.path.join(MT_Setup_object_PO.ResultsFolderSubTest, "BestModels",
                                     "weekday_%s.save" % (NameOfPredictor))):  # for weekday_weekend models
        indiv_predictor = indiv_model_onlypredict(indiv_splitter_instance=indiv_splitter(week_weekend_splitter),
                                                  Features_test=_X_test,
                                                  ResultsFolderSubTest=MT_Setup_object_PO.ResultsFolderSubTest,
                                                  NameOfPredictor=NameOfPredictor,
                                                  Recursive=MT_Setup_object_PO.OnlyPredictRecursive)
        Predicted = indiv_predictor.main()
        IndividualModel = "weekend/weekday"
        # Predicted = individual_model_week_weekend_onlypredict(_X_test, ResultsFolderSubTest, NameOfPredictor, OnlyPredictRecursive)
    elif os.path.isfile(os.path.join(MT_Setup_object_PO.ResultsFolderSubTest, "BestModels",
                                     "above_%s.save" % (NameOfPredictor))):  # for byFeature models
        byFeaturesplitter = byfeature_splitter(MT_Setup_object_PO.IndivThreshold, MT_Setup_object_PO.IndivFeature,
                                               _X_test)
        indiv_predictor = indiv_model_onlypredict(indiv_splitter_instance=indiv_splitter(byFeaturesplitter.splitter),
                                                  Features_test=_X_test,
                                                  ResultsFolderSubTest=MT_Setup_object_PO.ResultsFolderSubTest,
                                                  NameOfPredictor=NameOfPredictor,
                                                  Recursive=MT_Setup_object_PO.OnlyPredictRecursive)  # Todo: best models "byfeature" auch mit feature und threshold im namen abspeichern oder irgendwie damit das predicten unabhängig von den aktuellen werten von indivFeature usw. ist
        Predicted = indiv_predictor.main()
        IndividualModel = "byFeature"
    else:
        return False, False

    return Predicted, IndividualModel


def iterative_evaluation(MT_Setup_Object_PO, TestData, Model, horizon,
                         NameOfPredictor):  # horizon= amount of samples to predict in the future
    'Does an special evaluation which iteratively scores a period with the length of horizon in the whole period of TunedData. It returns the list of scores'
    # Todo: think of inserting this "iterative evaluation" to the regular scoring while training and testing(not only for onlypredict)
    n_folds = len(TestData) / horizon  # get how many times the horizon fits into the data
    n_folds = int(n_folds)  # cut of incomplete horizons
    # TunedData.index = range(len(TunedData)) #give them dataframe an counter index
    (TestData_X, TestData_Y) = SVF.split_signal_and_features(MT_Setup_Object_PO.NameOfSignal, TestData)

    if os.path.isfile(os.path.join(MT_Setup_Object_PO.ResultsFolder, "ScalerTracker.save")):  # if scaler was used
        ScaleTracker_Signal = joblib.load(
            os.path.join(MT_Setup_Object_PO.ResultsFolder, "ScalerTracker.save"))  # load used scaler

    fold_list = []
    for i in range(n_folds):
        measured_fold = TestData_Y[(horizon * i):(horizon * (i + 1))]
        Fold = TestData_X[(horizon * i):(horizon * (i + 1))]
        predicted_fold, Nothing = Model(MT_Setup_Object_PO, NameOfPredictor, Fold)  # predict on that fold
        # rescale
        predicted_fold = ScaleTracker_Signal.inverse_transform(SVF.reshape(predicted_fold))
        measured_fold = ScaleTracker_Signal.inverse_transform(SVF.reshape(measured_fold))

        fold_list.append([measured_fold, predicted_fold])
    return fold_list


def mean_scoring(fold_list, errormetric):  # processes the list of scores from "iterative_evaluation"
    'The list of scores (fold_list) is processed and the mean of all scores and the standard deviation over all scores are computed'
    errorlist = []
    for i in range(len(fold_list)):
        score = errormetric(fold_list[i][0], fold_list[i][1])
        errorlist.append(score)
    mean_scores = statistics.mean(errorlist)  # mean of all scores
    SD_scores = statistics.pstdev(errorlist)  # standard deviation of all scores
    return mean_scores, SD_scores, errorlist, errormetric


def main_OnlyPredict(MT_Setup_object_PO):
    print("Start only predicting: %s/%s/%s" % (
        MT_Setup_object_PO.NameOfData, MT_Setup_object_PO.NameOfExperiment, MT_Setup_object_PO.NameOfSubTest))
    _X_train, _Y_train, _X_test, _Y_test, Indexer, Data = MT.pre_handling(MT_Setup_object_PO, True)

    OnlyPredictFolder = os.path.join(MT_Setup_object_PO.ResultsFolderSubTest, "OnlyPredict",
                                     MT_Setup_object_PO.NameOfOnlyPredict)
    MT_Setup_object_PO.OnlyPredictFolder = OnlyPredictFolder

    # check if predict results are saved in the right folder:
    SVF.delete_and_create_folder(MT_Setup_object_PO.OnlyPredictFolder)

    AvailablePredictors = ["svr_bayesian_predictor", "rf_predictor", "ann_bayesian_predictor",
                           "gradientboost_bayesian", "lasso_bayesian", "svr_grid_search_predictor",
                           "gradientboost_gridsearch", "lasso_grid_search_predictor",
                           "ann_grid_search_predictor"]

    MT_RR_object_PO = MTRR()
    ModelSummaryObjectList = [MT_RR_object_PO.SVR_Summary, MT_RR_object_PO.RF_Summary, MT_RR_object_PO.ANN_Summary,
                              MT_RR_object_PO.GB_Summary, MT_RR_object_PO.Lasso_Summary,
                              MT_RR_object_PO.SVR_grid_Summary,
                              MT_RR_object_PO.GB_grid_Summary, MT_RR_object_PO.Lasso_grid_Summary,
                              MT_RR_object_PO.ANN_grid_Summary]

    for NameOfPredictor, ModelSummaryObject in zip(AvailablePredictors, ModelSummaryObjectList):
        only_predict(MT_Setup_object_PO, ModelSummaryObject, NameOfPredictor, _X_test, _Y_test, Indexer, Data)

    print("Finish only predicting : %s/%s/%s" % (
        MT_Setup_object_PO.NameOfData, MT_Setup_object_PO.NameOfExperiment, MT_Setup_object_PO.NameOfSubTest))
    print("________________________________________________________________________\n")
    print("________________________________________________________________________\n")
    MT_RR_object_PO.store_results(MT_Setup_object_PO)


if __name__ == "__main__":
    DT_Setup_object_PO, MT_Setup_object_PO = SVF.setup_object_initializer()
    main_OnlyPredict(MT_Setup_object_PO)

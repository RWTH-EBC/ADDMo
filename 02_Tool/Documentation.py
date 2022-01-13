import os
import shutil
import pandas as pd
import numpy as np
import sys

from openpyxl import load_workbook
from sklearn_pandas import DataFrameMapper
from PredictorDefinitions import *
from Functions.ErrorMetrics import *
from Functions.PlotFcn import *
from sklearn.feature_selection import mutual_info_regression, f_regression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from math import log

import SharedVariablesFunctions as SVF


def documentation_DataTuning(DT_Setup_object, timestart, timeend):
    print("Documentation")
    # dump the name of signal in the resultsfolder, so that you can always be pulled whenever you want to come back to that specific "Final Input Data"
    joblib.dump(DT_Setup_object.NameOfSignal, os.path.join(DT_Setup_object.ResultsFolder, "NameOfSignal.save"))

    # saving the methodology of creating FinalInputData in the ExcelFile "Settings"
    DfMethodology = pd.DataFrame(index=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                                 columns=["GlobalVariables", "ImportData", "Preprocessing", "PeriodSelection",
                                          "FeatureConstruction", "FeatureSelection"])
    # adding information to the dataframe
    DfMethodology.at[1, "GlobalVariables"] = "NameOfData = %s" % DT_Setup_object.NameOfData
    DfMethodology.at[2, "GlobalVariables"] = "NameOfExperiment = %s" % DT_Setup_object.NameOfExperiment
    DfMethodology.at[3, "GlobalVariables"] = "NameOfSignal = %s" % DT_Setup_object.NameOfSignal
    DfMethodology.at[4, "GlobalVariables"] = "Model used for Wrappers = %s" % DT_Setup_object.EstimatorWrapper.__name__
    DfMethodology.at[5, "GlobalVariables"] = "Parameter used for Wrappers = %s" % DT_Setup_object.WrapperParams
    DfMethodology.at[6, "GlobalVariables"] = "MinIncrease for Wrappers = %s" % DT_Setup_object.MinIncrease
    DfMethodology.at[7, "GlobalVariables"] = "Pipeline took %s seconds" % (timeend - timestart)

    DfMethodology.at[1, "Preprocessing"] = "How to deal NaN´s = %s" % DT_Setup_object.NaNDealing
    DfMethodology.at[2, "Preprocessing"] = "Initial feature select = %s" % DT_Setup_object.InitManFeatureSelect
    if DT_Setup_object.InitManFeatureSelect == True:
        DfMethodology.at[3, "Preprocessing"] = "Features selected = %s" % DT_Setup_object.InitFeatures
    DfMethodology.at[4, "Preprocessing"] = "StandardScaler = %s" % DT_Setup_object.StandardScaling
    DfMethodology.at[5, "Preprocessing"] = "RobustScaler = %s" % DT_Setup_object.RobustScaling
    DfMethodology.at[6, "Preprocessing"] = "NoScaling = %s" % DT_Setup_object.NoScaling
    DfMethodology.at[7, "Preprocessing"] = "Resample = %s" % DT_Setup_object.Resample
    if DT_Setup_object.Resample == True:
        DfMethodology.at[8, "Preprocessing"] = "Resolution = %s" % DT_Setup_object.Resolution
        DfMethodology.at[9, "Preprocessing"] = "WayOfResampling = %s" % DT_Setup_object.WayOfResampling

    DfMethodology.at[1, "PeriodSelection"] = "ManualSelection = %s" % DT_Setup_object.ManSelect
    if DT_Setup_object.ManSelect == True:
        DfMethodology.at[2, "PeriodSelection"] = "%s till %s" % (DT_Setup_object.StartDate, DT_Setup_object.EndDate)
    DfMethodology.at[3, "PeriodSelection"] = "TimeSeriesPlot = %s" % DT_Setup_object.TimeSeriesPlot

    DfMethodology.at[
        1, "FeatureConstruction"] = "Cross, auto, cloud correlation plot= %s" % DT_Setup_object.Cross_auto_cloud_correlation_plotting
    if DT_Setup_object.Cross_auto_cloud_correlation_plotting == True:
        DfMethodology.at[2, "FeatureConstruction"] = "LagsToBePlotted= %s" % DT_Setup_object.LagsToBePlotted
    DfMethodology.at[3, "FeatureConstruction"] = "DifferenceCreate= %s" % DT_Setup_object.DifferenceCreate
    if DT_Setup_object.DifferenceCreate == True:
        Word = "All" if DT_Setup_object.FeaturesDifference == True else DT_Setup_object.FeaturesDifference
        DfMethodology.at[4, "FeatureConstruction"] = "FeaturesToCreateDifference= %s" % Word
    DfMethodology.at[5, "FeatureConstruction"] = "Manual creation of OwnLags= %s" % DT_Setup_object.ManOwnlagCreate
    if DT_Setup_object.ManOwnlagCreate == True:
        DfMethodology.at[6, "FeatureConstruction"] = "OwnLags= %s" % DT_Setup_object.OwnLag
    DfMethodology.at[
        7, "FeatureConstruction"] = "Manual creation of FeatureLags= %s" % DT_Setup_object.ManFeaturelagCreate
    if DT_Setup_object.ManFeaturelagCreate == True:
        DfMethodology.at[8, "FeatureConstruction"] = "FeatureLags= %s" % DT_Setup_object.FeatureLag
    DfMethodology.at[
        9, "FeatureConstruction"] = "Automatic creation of time series ownlags= %s" % DT_Setup_object.AutomaticTimeSeriesOwnlagConstruct
    if DT_Setup_object.AutomaticTimeSeriesOwnlagConstruct == True:
        DfMethodology.at[10, "FeatureConstruction"] = "Minimal Ownlag= %s" % DT_Setup_object.MinOwnLag
    DfMethodology.at[
        11, "FeatureConstruction"] = "Automatic creation of lagged features= %s" % DT_Setup_object.AutoFeaturelagCreate
    if DT_Setup_object.AutoFeaturelagCreate == True:
        DfMethodology.at[12, "FeatureConstruction"] = "First lag to be considered= %s" % DT_Setup_object.MinFeatureLag
        DfMethodology.at[13, "FeatureConstruction"] = "Last lag to be considered= %s" % DT_Setup_object.MaxFeatureLag

    DfMethodology.at[1, "FeatureSelection"] = "Manual feature selection = %s" % DT_Setup_object.ManFeatureSelect
    if DT_Setup_object.ManFeatureSelect == True:
        DfMethodology.at[2, "FeatureSelection"] = "Selected Features= %s" % DT_Setup_object.FeatureSelect
    DfMethodology.at[3, "FeatureSelection"] = "Low Variance Filter = %s" % DT_Setup_object.LowVarianceFilter
    if DT_Setup_object.LowVarianceFilter == True:
        DfMethodology.at[4, "FeatureSelection"] = "Threshold Variance= %s" % DT_Setup_object.Threshold_LowVarianceFilter
    DfMethodology.at[5, "FeatureSelection"] = "Independent Component Analysis = %s" % DT_Setup_object.ICA
    DfMethodology.at[6, "FeatureSelection"] = "Univariate Filter = %s" % DT_Setup_object.UnivariateFilter
    if DT_Setup_object.UnivariateFilter == True:
        DfMethodology.at[7, "FeatureSelection"] = "Score function= %s" % DT_Setup_object.Score_func
        DfMethodology.at[8, "FeatureSelection"] = "Search mode= %s" % DT_Setup_object.SearchMode
        DfMethodology.at[
            9, "FeatureSelection"] = "Search mode threshold parameter= %s" % DT_Setup_object.Param_univariate_filter
    DfMethodology.at[10, "FeatureSelection"] = "Embedded-Recursive Feature Selection = %s" % (
            DT_Setup_object.RecursiveFeatureSelection or DT_Setup_object.EmbeddedFeatureSelectionThreshold)
    if (DT_Setup_object.RecursiveFeatureSelection or DT_Setup_object.EmbeddedFeatureSelectionThreshold) == True:
        DfMethodology.at[11, "FeatureSelection"] = "Embedded Estimator = %s" % DT_Setup_object.EstimatorEmbedded
        if DT_Setup_object.RecursiveFeatureSelection == True:
            DfMethodology.at[
                12, "FeatureSelection"] = "Number of Features to select= %s" % DT_Setup_object.N_feature_to_select_RFE
            if DT_Setup_object.N_feature_to_select_RFE == "automatic":
                DfMethodology.at[13, "FeatureSelection"] = "CrossValidation= %s" % DT_Setup_object.CV_DT
            else:
                DfMethodology.at[14, "FeatureSelection"] = "CrossValidation= None"
        if DT_Setup_object.EmbeddedFeatureSelectionThreshold == True:
            DfMethodology.at[
                15, "FeatureSelection"] = "Feature importance threshold = %s" % DT_Setup_object.Threshold_embedded
            DfMethodology.at[16, "FeatureSelection"] = "CrossValidation= None"

    # save this dataframe in an excel
    ExcelFile = os.path.join(DT_Setup_object.ResultsFolder, "Settings_%s.xlsx" % (DT_Setup_object.NameOfExperiment))
    writer = pd.ExcelWriter(ExcelFile, engine="openpyxl")
    DfMethodology.to_excel(writer, sheet_name="Methodology")
    writer.save()
    writer.close()


def documentation_model_tuning(MT_Setup_Object, RR_Model_Summary, NameOfPredictor, Y_Predicted, Y_test, Y_train,
                  ComputationTime, Scores, HyperparameterGrid=None, Bestparams=None, IndividualModel="",
                  FeatureImportance="Not available"):
    (R2, STD, RMSE, MAPE, MAE) = Scores

    # save summary of setup and evaluation
    dfSummary = pd.DataFrame(index=[0])
    dfSummary['Estimator'] = NameOfPredictor
    if Y_train is not None:  # don´t document this if "onlypredict" is used
        dfSummary['Start_date_Fit'] = MT_Setup_Object.StartTraining
        dfSummary['End_date_Fit'] = MT_Setup_Object.EndTraining
    dfSummary['Start_date_Predict'] = MT_Setup_Object.StartTesting
    dfSummary['End_date_Predict'] = MT_Setup_Object.EndTesting
    if Y_train is not None:  # don´t document this if "onlypredict" is used
        dfSummary['Total Train Samples'] = len(Y_train.index)
    dfSummary['Test Samples'] = len(Y_test.index)
    dfSummary['Recursive'] = MT_Setup_Object.GlobalRecu
    dfSummary['Shuffle'] = MT_Setup_Object.GlobalShuffle
    if HyperparameterGrid is not None:
        dfSummary['Range Hyperparameter'] = str(HyperparameterGrid)
        dfSummary['CrossValidation'] = str(MT_Setup_Object.GlobalCV_MT)
        dfSummary['Best Hyperparameter'] = str(Bestparams)
        if MT_Setup_Object.GlobalMaxEval_HyParaTuning is not None:
            dfSummary['Max Bayesian Evaluations'] = str(MT_Setup_Object.GlobalMaxEval_HyParaTuning)
    dfSummary["Feature importance"] = str(FeatureImportance)
    dfSummary['Individual model'] = IndividualModel
    if IndividualModel == "byFeature":
        dfSummary['IndivFeature'] = MT_Setup_Object.IndivFeature
        dfSummary['IndivThreshold'] = MT_Setup_Object.IndivThreshold
    dfSummary['Eval_R2'] = R2
    dfSummary['Eval_RMSE'] = RMSE
    dfSummary['Eval_MAPE'] = MAPE
    dfSummary['Eval_MAE'] = MAE
    dfSummary['Standard deviation'] = STD
    dfSummary['Computation Time'] = "%.2f seconds" % ComputationTime
    dfSummary = dfSummary.T
    # write summary of setup and evaluation in excel File
    SummaryFile = os.path.join(MT_Setup_Object.ResultsFolderSubTest,
                               "Summary_%s_%s.xlsx" % (NameOfPredictor, MT_Setup_Object.NameOfSubTest))
    writer = pd.ExcelWriter(SummaryFile)
    dfSummary.to_excel(writer, float_format='%.6f')
    writer.save()

    # export prediction to Excel
    SaveFileName_excel = os.path.join(MT_Setup_Object.ResultsFolderSubTest,
                                      "Prediction_%s_%s.xlsx" % (NameOfPredictor, MT_Setup_Object.NameOfSubTest))
    Y_Predicted.to_frame(name=MT_Setup_Object.NameOfSignal).to_excel(SaveFileName_excel)

    # save model tuning runtime results in ModelTuningRuntimeResults class object

    RR_Model_Summary.model_name = NameOfPredictor
    if Y_train is not None:
        RR_Model_Summary.total_train_samples = len(Y_train.index)
    RR_Model_Summary.test_samples = len(Y_test.index)
    if HyperparameterGrid is not None:
        RR_Model_Summary.best_hyperparameter = str(Bestparams)
    RR_Model_Summary.feature_importance = str(FeatureImportance)
    RR_Model_Summary.eval_R2 = R2
    RR_Model_Summary.eval_RMSE = RMSE
    RR_Model_Summary.eval_MAPE = MAPE
    RR_Model_Summary.eval_MAE = MAE
    RR_Model_Summary.standard_deviation = STD
    RR_Model_Summary.computation_time = "%.2f seconds" % ComputationTime

    # return Score for modelselection
    return R2

def documentation_only_predict(MT_Setup_Object_PO, RR_Model_Summary, NameOfPredictor, Y_Predicted, Y_test,
                  ComputationTime, Scores, IndividualModel="", FeatureImportance="Not available"):
    (R2, STD, RMSE, MAPE, MAE) = Scores

    # save summary of setup and evaluation
    dfSummary = pd.DataFrame(index=[0])
    dfSummary['Estimator'] = NameOfPredictor
    dfSummary['Start_date_Predict'] = MT_Setup_Object_PO.StartTesting
    dfSummary['End_date_Predict'] = MT_Setup_Object_PO.EndTesting
    dfSummary['Test Samples'] = len(Y_test.index)
    dfSummary['Recursive'] = MT_Setup_Object_PO.OnlyPredictRecursive
    dfSummary['Shuffle'] = None
    dfSummary["Feature importance"] = str(FeatureImportance)
    dfSummary['Individual model'] = IndividualModel
    if IndividualModel == "byFeature":
        dfSummary['IndivFeature'] = MT_Setup_Object_PO.IndivFeature
        dfSummary['IndivThreshold'] = MT_Setup_Object_PO.IndivThreshold
    dfSummary['Eval_R2'] = R2
    dfSummary['Eval_RMSE'] = RMSE
    dfSummary['Eval_MAPE'] = MAPE
    dfSummary['Eval_MAE'] = MAE
    dfSummary['Standard deviation'] = STD
    dfSummary['Computation Time'] = "%.2f seconds" % ComputationTime
    dfSummary = dfSummary.T
    # write summary of setup and evaluation in excel File
    SummaryFile = os.path.join(MT_Setup_Object_PO.OnlyPredictFolder,
                               "Summary_%s_%s.xlsx" % (NameOfPredictor, MT_Setup_Object_PO.NameOfSubTest))
    writer = pd.ExcelWriter(SummaryFile)
    dfSummary.to_excel(writer, float_format='%.6f')
    writer.save()

    # export prediction to Excel
    SaveFileName_excel = os.path.join(MT_Setup_Object_PO.OnlyPredictFolder,
                                      "Prediction_%s_%s.xlsx" % (NameOfPredictor, MT_Setup_Object_PO.NameOfSubTest))
    Y_Predicted.to_frame(name=MT_Setup_Object_PO.NameOfSignal).to_excel(SaveFileName_excel)

    # save model tuning runtime results in ModelTuningRuntimeResults class object

    RR_Model_Summary.model_name = NameOfPredictor
    RR_Model_Summary.test_samples = len(Y_test.index)
    RR_Model_Summary.feature_importance = str(FeatureImportance)
    RR_Model_Summary.eval_R2 = R2
    RR_Model_Summary.eval_RMSE = RMSE
    RR_Model_Summary.eval_MAPE = MAPE
    RR_Model_Summary.eval_MAE = MAE
    RR_Model_Summary.standard_deviation = STD
    RR_Model_Summary.computation_time = "%.2f seconds" % ComputationTime

def visualization(MT_Setup_Object, NameOfPredictor, prediction, measurement, Scores):
    (R2, STD, RMSE, MAPE, MAE) = Scores

    plot_predict_measured(prediction,
                          measurement,
                          MAE=MAE, R2=R2,
                          StartDatePredict=MT_Setup_Object.StartTesting,
                          SavePath=MT_Setup_Object.ResultsFolderSubTest,
                          nameOfSignal=MT_Setup_Object.NameOfSignal,
                          BlackBox=NameOfPredictor,
                          NameOfSubTest=MT_Setup_Object.NameOfSubTest)

    plot_Residues(prediction,
                  measurement,
                  MAE=MAE, R2=R2,
                  SavePath=MT_Setup_Object.ResultsFolderSubTest,
                  nameOfSignal=MT_Setup_Object.NameOfSignal,
                  BlackBox=NameOfPredictor,
                  NameOfSubTest=MT_Setup_Object.NameOfSubTest)


# OP
def documentation_iterative_evaluation(MT_Setup_object_PO, NameOfPredictor, mean_score, SD_score, errorlist, horizon,
                                      errormetric):
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

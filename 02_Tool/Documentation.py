import os
import shutil
import pandas as pd
import numpy as np
import sys
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


def documentation(MT_Setup_Object, RR_Model_Summary, NameOfPredictor, Y_Predicted, Y_test, Y_train,
                  ComputationTime, Scores, HyperparameterGrid=None, Bestparams=None, IndividualModel="",
                  FeatureImportance="Not available", ):
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

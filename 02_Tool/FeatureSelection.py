import os
import time

import pandas as pd
import numpy as np
from pandas.io.excel import ExcelWriter

from openpyxl import load_workbook
import sys
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectFromModel
from FeatureConstruction import automatic_timeseries_ownlag_constructor
from sklearn.decomposition import FastICA
from sklearn.feature_selection import GenericUnivariateSelect
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split

# from GlobalVariables import *
import SharedVariablesFunctions as SVF


# Todo: add function to just delete certain features
# Manual Selection of Features (By the columns of the "FeatureConstruction" Excel Table)
def man_feature_select(DT_Setup_object, Data):
    if not DT_Setup_object.ColumnOfSignal in DT_Setup_object.FeatureSelect:  # keep the column of signal
        FeatureSelect = np.append(DT_Setup_object.FeatureSelect,
                                  DT_Setup_object.ColumnOfSignal)  # add the column of signal to the features which shall be kept
    Data = Data[Data.columns[FeatureSelect]]  # select only those columns which shall be kept
    return (Data)


# Pre-Filter removing features with low variance
def low_variance_filter(DT_Setup_object, Data):
    (X, Y) = SVF.split_signal_and_features(DT_Setup_object.NameOfSignal, Data=Data)
    filter = VarianceThreshold(threshold=DT_Setup_object.Threshold_LowVarianceFilter)  # set filter
    filter = filter.fit(X=X)  # train filter
    Features_transformed = filter.transform(X=X)  # transform the data
    Data = SVF.merge_signal_and_features_embedded(DT_Setup_object.NameOfSignal, X_Data=X, Y_Data=Y, support=filter.get_support(indices=True),
                                                  X_Data_transformed=Features_transformed)
    return Data


# Filter Independent Component Analysis (ICA)
def filter_ica(DT_Setup_object, Data):
    (X, Y) = SVF.split_signal_and_features(DT_Setup_object.NameOfSignal, Data=Data)
    Ica = FastICA(max_iter=1000)
    Features_transformed = Ica.fit_transform(X=X)
    Data = SVF.merge_signal_and_features(DT_Setup_object.NameOfSignal, X_Data=X, Y_Data=Y, X_Data_transformed=Features_transformed)
    return Data


# Filter Univariate with scoring function f-test or mutual information and search mode : {‘percentile’, ‘k_best’, ‘fpr’, ‘fdr’, ‘fwe’}
def filter_univariate(DT_Setup_object, Data):
    (X, Y) = SVF.split_signal_and_features(DT_Setup_object.NameOfSignal, Data=Data)
    filter = GenericUnivariateSelect(score_func=DT_Setup_object.Score_func, mode=DT_Setup_object.SearchMode,
                                     param=DT_Setup_object.Param_univariate_filter)
    filter = filter.fit(X=X, y=Y)
    Features_transformed = filter.transform(X=X)
    Data = SVF.merge_signal_and_features_embedded(DT_Setup_object.NameOfSignal, X_Data=X, Y_Data=Y, support=filter.get_support(indices=True),
                                                  X_Data_transformed=Features_transformed)
    return Data


# embedded Feature Selection by recursive feature elemination (Feature Subset Selection, multivariate)
def embedded__recursive_feature_selection(DT_Setup_object, Data):
    (X, Y) = SVF.split_signal_and_features(DT_Setup_object.NameOfSignal, Data=Data)
    # split into automatic and selection by number because those are two different functions
    if DT_Setup_object.N_features_to_select == "automatic":
        selector = RFECV(estimator=DT_Setup_object.EstimatorEmbedded, step=1, cv=DT_Setup_object.CV_DT)
        selector = selector.fit(X, Y)
        print("Ranks of all Features %s" % selector.ranking_)
        Features_transformed = selector.transform(X)
    else:
        selector = RFE(estimator=DT_Setup_object.EstimatorEmbedded,
                       n_features_to_select=DT_Setup_object.N_feature_to_select_RFE, step=1)
        selector = selector.fit(X, Y)
        print("Ranks of all Features %s" % selector.ranking_)
        Features_transformed = selector.transform(X)
    Data = SVF.merge_signal_and_features_embedded(DT_Setup_object.NameOfSignal, X_Data=X, Y_Data=Y, support=selector.get_support(indices=True),
                                                  X_Data_transformed=Features_transformed)
    return Data


# embedded Feature Selection by importance with setting an threshold of importance (Feature Selection through ranking; univariate)
def embedded__feature_selection_by_importance_threshold(DT_Setup_object, Data):
    (X, Y) = SVF.split_signal_and_features(DT_Setup_object.NameOfSignal, Data)
    Estimator = DT_Setup_object.EstimatorEmbedded.fit(X, Y)
    # Estimator.feature_importances_ #Todo: delete if proven unnecessary
    print("Importance of all Features %s" % Estimator.feature_importances_)
    selector = SelectFromModel(threshold=DT_Setup_object.Threshold_embedded, estimator=Estimator, prefit=True)
    Features_transformed = selector.transform(X)
    Data = SVF.merge_signal_and_features_embedded(DT_Setup_object.NameOfSignal, X_Data=X, Y_Data=Y, support=selector.get_support(indices=True),
                                                  X_Data_transformed=Features_transformed)
    return Data


def wrapper__recursive_feature_selection(DT_Setup_object, Data):
    print("recursive feature selection via wrapper START")
    (X, Y) = SVF.split_signal_and_features(DT_Setup_object.NameOfSignal, Data)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)
    Result_dic = DT_Setup_object.EstimatorWrapper(X_train, y_train, X_test, y_test,
                                                  *DT_Setup_object.WrapperParams)  # score will be done over hold out 0.25 percent of data
    Score = Result_dic["score"]  # get the score  #get initial score
    Score_i = Score
    while True:  # loop as long as deleting features increases accuracy
        Score = Score_i  # set score equal to the new and better score_i
        (X_i, Y) = SVF.split_signal_and_features(DT_Setup_object.NameOfSignal, Data)
        for column in X_i:  # loop through all columns
            X_ii = X_i.drop(column, axis=1)  # drop the respective columns
            X_train_ii, X_test_ii, y_train, y_test = train_test_split(X_ii, Y, test_size=0.25)
            Result_dic = DT_Setup_object.EstimatorWrapper(X_train_ii, y_train, X_test_ii, y_test,
                                                          *DT_Setup_object.WrapperParams)  # score will be done over hold out 0.25 percent of data
            Score_ii = Result_dic["score"]  # get the score
            if Score_ii > Score_i:  # check for the data that provided the best score
                Score_i = Score_ii
                Todrop = column  # get the column that should be dropped
        if Score_i > (
                Score + DT_Setup_object.MinIncrease):  # is new score higher than the old one? (take care that >= would not work, since in the case that no new score was set, score_i is equal to score
            Data = Data.drop(Todrop, axis=1)
            print("Dropped column: %s" % Todrop)
        else:
            break
    return Data


# Main#############################################################
def main(DT_Setup_object, DT_RR_object):
    print("FeatureSelection")
    startTime = time.time()

    Data = DT_RR_object.df_feature_construction_data
    Data_AllSamples = DT_RR_object.df_preprocessing_data

    if DT_Setup_object.ManFeatureSelect == True:
        Data = man_feature_select(DT_Setup_object, Data)
    if DT_Setup_object.LowVarianceFilter == True:
        Data = low_variance_filter(DT_Setup_object, Data)
    if DT_Setup_object.ICA == True:
        Data = filter_ica(DT_Setup_object, Data)
    if DT_Setup_object.UnivariateFilter == True:
        Data = filter_univariate(DT_Setup_object, Data)
    if DT_Setup_object.EmbeddedFeatureSelectionThreshold == True:
        Data = embedded__feature_selection_by_importance_threshold(DT_Setup_object, Data)
    if DT_Setup_object.RecursiveFeatureSelection == True:
        Data = embedded__recursive_feature_selection(DT_Setup_object, Data)
    if DT_Setup_object.WrapperRecursiveFeatureSelection == True:
        Data = wrapper__recursive_feature_selection(DT_Setup_object, Data)
    if DT_Setup_object.AutomaticTimeSeriesOwnlagConstruct == True:  # method from FeatureConstruction
        Data = automatic_timeseries_ownlag_constructor(DT_Setup_object, Data, Data_AllSamples)

    endTime = time.time()
    DT_RR_object.feature_selection_time = endTime - startTime
    DT_RR_object.df_feature_selection_data = Data

    #  save dataframe in an pickle
    Data.to_pickle(os.path.join(DT_Setup_object.PathToPickles, "ThePickle_from_FeatureSelection" + '.pickle'))

    # save dataframe in the ProcessedInputData excel file
    ExcelFile = os.path.join(DT_Setup_object.ResultsFolder,
                             "ProcessedInputData_%s.xlsx" % (DT_Setup_object.NameOfExperiment))
    book = load_workbook(ExcelFile)
    writer = pd.ExcelWriter(ExcelFile, engine="openpyxl")
    writer.book = book
    writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
    Data.to_excel(writer, sheet_name="FeatureSelection")
    writer.save()
    writer.close()

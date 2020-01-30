import os
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

#from GlobalVariables import *
import SharedVariables as SV

#Todo: add function to just delete certain features
#Manual Selection of Features (By the columns of the "FeatureConstruction" Excel Table)
def man_feature_select(FeatureSelect, Data):
    if not SV.ColumnOfSignal in FeatureSelect:  # keep the column of signal
        FeatureSelect = np.append(FeatureSelect, SV.ColumnOfSignal) #add the column of signal to the features which shall be kept
    Data = Data[Data.columns[FeatureSelect]] #select only those columns which shall be kept
    return (Data)

#Pre-Filter removing features with low variance
def low_variance_filter(Data, Threshold_LowVarianceFilter):
    (X, Y) = SV.split_signal_and_features(Data=Data)
    filter = VarianceThreshold(threshold=Threshold_LowVarianceFilter)#set filter
    filter = filter.fit(X=X) #train filter
    Features_transformed = filter.transform(X=X) #transform the data
    Data = SV.merge_signal_and_features_embedded(X_Data=X, Y_Data=Y, support=filter.get_support(indices=True),
                                              X_Data_transformed=Features_transformed)
    return Data

#Filter Independent Component Analysis (ICA)
def filter_ica(Data):
    (X, Y) = SV.split_signal_and_features(Data=Data)
    Ica = FastICA(max_iter=1000)
    Features_transformed = Ica.fit_transform(X=X)
    Data = SV.merge_signal_and_features(X_Data=X, Y_Data=Y, X_Data_transformed=Features_transformed)
    return Data

#Filter Univariate with scoring function f-test or mutual information and search mode : {‘percentile’, ‘k_best’, ‘fpr’, ‘fdr’, ‘fwe’}
def filter_univariate(Data, Score_func, SearchMode, Param):
    (X, Y) = SV.split_signal_and_features(Data=Data)
    filter = GenericUnivariateSelect(score_func=Score_func, mode=SearchMode, param=Param)
    filter = filter.fit(X=X, y=Y)
    Features_transformed = filter.transform(X=X)
    Data = SV.merge_signal_and_features_embedded(X_Data=X, Y_Data=Y, support=filter.get_support(indices=True),
                                              X_Data_transformed=Features_transformed)
    return Data


#embedded Feature Selection by recursive feature elemination (Feature Subset Selection, multivariate)
def embedded__recursive_feature_selection(Data, Estimator, N_features_to_select, CV):
    (X, Y) = SV.split_signal_and_features(Data=Data)
    #split into automatic and selection by number because those are two different functions
    if N_features_to_select == "automatic":
        selector = RFECV(estimator=Estimator, step=1, cv=CV)
        selector = selector.fit(X, Y)
        print("Ranks of all Features %s" %selector.ranking_)
        Features_transformed = selector.transform(X)
    else:
        selector = RFE(estimator=Estimator, n_features_to_select=N_features_to_select, step=1)
        selector = selector.fit(X, Y)
        print("Ranks of all Features %s" %selector.ranking_)
        Features_transformed = selector.transform(X)
    Data = SV.merge_signal_and_features_embedded(X_Data=X, Y_Data=Y, support=selector.get_support(indices=True), X_Data_transformed=Features_transformed)
    return Data

#embedded Feature Selection by importance with setting an threshold of importance (Feature Selection through ranking; univariate)
def embedded__feature_selection_by_importance_threshold(Data, Estimator, Threshold_embedded):
    (X, Y) = SV.split_signal_and_features(Data=Data)
    Estimator = Estimator.fit(X, Y)
    #Estimator.feature_importances_ #Todo: delete if proven unnecessary
    print("Importance of all Features %s" %Estimator.feature_importances_)
    selector =SelectFromModel(threshold=Threshold_embedded, estimator=Estimator, prefit=True)
    Features_transformed = selector.transform(X)
    Data = SV.merge_signal_and_features_embedded(X_Data=X, Y_Data=Y, support=selector.get_support(indices=True), X_Data_transformed=Features_transformed)
    return Data

def wrapper__recursive_feature_selection(Data, Estimator, params,MinIncrease):
    print("recursive feature selection via wrapper START")
    (X, Y) = SV.split_signal_and_features(Data)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25)
    Result_dic = Estimator(X_train, y_train, X_test, y_test, *params)  # score will be done over hold out 0.25 percent of data
    Score = Result_dic["score"]  #get the score  #get initial score
    Score_i = Score
    while True: #loop as long as deleting features increases accuracy
        Score = Score_i #set score equal to the new and better score_i
        (X_i, Y) = SV.split_signal_and_features(Data)
        for column in X_i: #loop through all columns
            X_ii = X_i.drop(column, axis=1)#drop the respective columns
            X_train_ii, X_test_ii, y_train, y_test = train_test_split(X_ii, Y, test_size=0.25)
            Result_dic = Estimator(X_train_ii, y_train, X_test_ii, y_test,*params)  # score will be done over hold out 0.25 percent of data
            Score_ii = Result_dic["score"]#get the score
            if Score_ii > Score_i: #check for the data that provided the best score
                Score_i = Score_ii
                Todrop = column #get the column that should be dropped
        if Score_i > (Score+MinIncrease): #is new score higher than the old one? (take care that >= would not work, since in the case that no new score was set, score_i is equal to score
            Data = Data.drop(Todrop, axis=1)
            print("Dropped column: %s" %Todrop)
        else:
            break
    return Data

#Main#############################################################
def main():
    print("FeatureSelection")

    #read in the pickle produced by "FeatureConstruction"
    Data = pd.read_pickle(os.path.join(SV.PathToPickles, "ThePickle_from_FeatureConstruction" + '.pickle'))

    if SV.ManFeatureSelect == True:
        Data = man_feature_select(SV.FeatureSelect, Data)
    if SV.LowVarianceFilter == True:
        Data = low_variance_filter(Data, SV.Threshold_LowVarianceFilter)
    if SV.ICA == True:
        Data = filter_ica(Data)
    if SV.UnivariateFilter == True:
        Data =filter_univariate(Data, SV.Score_func, SV.SearchMode, SV.Param_univariate_filter)
    if SV.EmbeddedFeatureSelectionThreshold == True:
        Data = embedded__feature_selection_by_importance_threshold(Data=Data, Estimator=SV.EstimatorEmbedded, Threshold_embedded=SV.Threshold_embedded)
    if SV.RecursiveFeatureSelection == True:
        Data = embedded__recursive_feature_selection(Data=Data, Estimator=SV.EstimatorEmbedded, N_features_to_select=SV.N_feature_to_select_RFE, CV=SV.CV_DT)
    if SV.WrapperRecursiveFeatureSelection == True:
        Data = wrapper__recursive_feature_selection(Data=Data, Estimator=SV.EstimatorWrapper, params=SV.WrapperParams, MinIncrease=SV.MinIncrease)
    if SV.AutomaticTimeSeriesOwnlagConstruct == True: #method from FeatureConstruction
        Data = automatic_timeseries_ownlag_constructor(Data=Data, Data_AllSamples=pd.read_pickle(os.path.join(SV.PathToPickles, "ThePickle_from_Preprocessing" + '.pickle')), MinOwnLag=SV.MinOwnLag, Estimator=SV.EstimatorWrapper, params=SV.WrapperParams, MinIncrease=SV.MinIncrease)

    #  save dataframe in an pickle
    Data.to_pickle(os.path.join(SV.PathToPickles, "ThePickle_from_FeatureSelection" + '.pickle'))

    # save dataframe in the ProcessedInputData excel file
    ExcelFile = os.path.join(SV.ResultsFolder, "ProcessedInputData_%s.xlsx"%(SV.NameOfExperiment))
    book = load_workbook(ExcelFile)
    writer = pd.ExcelWriter(ExcelFile, engine="openpyxl")
    writer.book = book
    writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
    Data.to_excel(writer, sheet_name="FeatureSelection")
    writer.save()
    writer.close()


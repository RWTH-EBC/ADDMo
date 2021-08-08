import time
import sys
import os

import pandas as pd

from openpyxl import load_workbook
from Functions.PlotFcn import *
from sklearn.model_selection import train_test_split
import SharedVariables as SV

# Todo: get the resolution of the data

# Cross, Cloud and Autocorrelation plots
def cross_auto_cloud_correlation_plotting(DT_Setup_object, Data):
    # set folder for sensor and resolution
    CorrelationResultsFolder = "%s/%s" % (DT_Setup_object.ResultsFolder, "cross_auto_cloud_plotting")
    # check if directory exists, if not, create it
    if not os.path.exists(CorrelationResultsFolder):
        os.makedirs(CorrelationResultsFolder)

    # plot simple dependency on influence factors
    Labels = [column for column in Data.columns]
    for i in range(1, len(Labels)):
        plot_x_y(Data[Labels[i]], Data[DT_Setup_object.NameOfSignal], CorrelationResultsFolder)

    # plot autocorrelation
    plot_acf(Data[DT_Setup_object.NameOfSignal], CorrelationResultsFolder, lags=DT_Setup_object.LagsToBePlotted)

    # make an array for correlation coefficients
    corrCoeffMax = np.zeros(len(Labels))
    DictCoerrCoeff = dict()

    # plot cross correlation between signal to be forecasted and exogenous input
    for i in range(1, len(Labels)):
        plot_crosscorr(Data[DT_Setup_object.NameOfSignal], Data[Labels[i]], CorrelationResultsFolder, lags=DT_Setup_object.LagsToBePlotted)
        corrCoeff = plt.xcorr(Data[DT_Setup_object.NameOfSignal], Data[Labels[i]], maxlags=DT_Setup_object.LagsToBePlotted, normed=True)
        # max value over the values of the correlation coefficient
        DictCoerrCoeff[Data[Labels[i]].name] = np.amax(corrCoeff[1])
        corrCoeffDf = pd.DataFrame(DictCoerrCoeff, index=[Data[DT_Setup_object.NameOfSignal].name], columns=Labels)

    # save cross correlations in an excel file
    ExcelFile = "%s/CrossCorrelationCoefficients.xlsx" % (CorrelationResultsFolder)
    writer = pd.ExcelWriter(ExcelFile)
    corrCoeffDf.to_excel(writer, sheet_name=DT_Setup_object.NameOfExperiment, float_format='%.2f')
    writer.save()


# Manual creation of lags of the Features(Signal excluded)
def manual_featurelag_create(DT_Setup_object, Data, Data_AllSamples):
    if len(DT_Setup_object.FeatureLag) != len(list(Data)):
        sys.exit("Your FeatureLag Array has to have as many Arrays as Columns of your input data(Index excluded)")
    for i in range(0, len(DT_Setup_object.FeatureLag)):
        NameOfFeature = SV.nameofmeter(Data, i)  # get the name of the meter in column i
        if NameOfFeature != DT_Setup_object.NameOfSignal:  # making sure this method does not produce OwnLags
            Xauto = Data_AllSamples[NameOfFeature]  # copy of the Feature to use for shifting
            for lag in range(0, len(DT_Setup_object.FeatureLag[i])):
                FeatureLagName = NameOfFeature + "_lag" + str(DT_Setup_object.FeatureLag[i][lag])  # create a column name per lag
                DataShift = Xauto.shift(periods=DT_Setup_object.FeatureLag[i][lag]).fillna(
                    method='bfill')  # shift with the respective values with the respective lag
                Data = pd.concat([Data, DataShift.rename(FeatureLagName)], axis=1,
                                 join="inner")  # joining the dataframes just for the selected period
    return (Data)


# Manual creation of lags of the Signal(OwnLags)
def man_ownlag_create(DT_Setup_object, Data, Data_AllSamples):
    # is there a relevant autocorrelation for the signal to be predicted?
    Xauto = Data_AllSamples[DT_Setup_object.NameOfSignal]  # copy of Y to use for shifting
    for lag in range(0, len(DT_Setup_object.OwnLag)):
        OwnLagName = DT_Setup_object.NameOfSignal + "_lag_" + str(DT_Setup_object.OwnLag[lag])  # create a column name per lag
        DataShift = Xauto.shift(periods=DT_Setup_object.OwnLag[lag]).fillna(method='bfill')  # shift with the lag to be considered
        Data = pd.concat([Data, DataShift.rename(OwnLagName)], axis=1, join="inner")  # joining the dataframes just for the selected period
    return (Data)


# Automatic creation of difference data through building the delta value of t - (t-1)
def difference_create(DT_Setup_object, Data):
    if DT_Setup_object.FeaturesDifference == True:  # for all features a derivative series should be constructed
        (X, Y) = SV.split_signal_and_features(Data)
    else:  # if certain features are selected
        (X, Y) = SV.split_signal_and_features(Data)
        X = X[X.columns[DT_Setup_object.FeaturesDifference]]  # select only those columns of which a difference shall be created
    for column in X:  # loop through all columns
        DifferenceName = "Delta_" + column  # construct a new name for the difference data
        Difference = X[column].diff()  # build difference of respective feature
        Difference = Difference.fillna(0)  # fill up first row with 0 (since the first row has no value before and hence should be 0)
        Difference = SV.post_scaler(Difference, DT_Setup_object.StandardScaling, DT_Setup_object.RobustScaling)
        Data = pd.concat([Data, Difference.rename(columns={0: DifferenceName})], axis=1,join="inner")  # joining the dataframes just for the selected period
    return Data

# automatic timeseries ownlag construction used in Feature Selection (Disadvantage: Only finds local optimum of the amount of Ownlags)
def automatic_timeseries_ownlag_constructor(DT_Setup_object, Data, Data_AllSamples):
    print("Auto timeseries-ownlag constructor START")
    Xauto = Data_AllSamples[DT_Setup_object.NameOfSignal]  # copy of Y to use for shifting
    (X, Y) = DT_Setup_object.split_signal_and_features(Data)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)
    Result_dic = DT_Setup_object.EstimatorWrapper(X_train, y_train, X_test, y_test,
                           *DT_Setup_object.WrapperParams)  # score will be done over hold out 0.25 percent of data
    Score = Result_dic["score"]  # get the score  #get initial score
    Score_i = Score
    i = DT_Setup_object.MinOwnLag - 1  # just for easier looping
    while True:  # loop as long as a new ownlag increases the accuracy
        i += 1
        Score = Score_i  # set score equal to the new and better score_i
        OwnLagName = DT_Setup_object.NameOfSignal + "_lag_" + str(i)  # create a column name per lag
        DataShift = Xauto.shift(periods=i).fillna(method='bfill')  # shift with the lag to be considered
        Data = pd.concat([Data, DataShift.rename(OwnLagName)], axis=1,
                         join="inner")  # joining the dataframes just for the selected period
        (X, Y) = SV.split_signal_and_features(Data)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)
        Result_dic = DT_Setup_object.EstimatorWrapper(X_train, y_train, X_test, y_test,
                               *DT_Setup_object.WrapperParams)  # score will be done over hold out 0.25 percent of data
        Score_i = Result_dic["score"]  # get the score with the additional ownlag
        if not (Score + DT_Setup_object.MinIncrease) <= Score_i:
            break
    Data = Data.drop([OwnLagName], axis=1)  # drop the last ownlag that was not improving the score
    return Data


def auto_featurelag_constructor(DT_Setup_object, Data, Data_AllSamples):
    print("auto featurelag constructor START")
    # wrapper gets default accuracy
    (X, Y) = SV.split_signal_and_features(Data)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)
    Result_dic = DT_Setup_object.EstimatorWrapper(X_train, y_train, X_test, y_test, *DT_Setup_object.WrapperParams)  # score will be done over hold out 0.25 percent of data
    Score = Result_dic["score"]  # get the score  #get initial score
    Columns = list(Data)
    Data_c = Data  # copy of Data that is never changed
    for i in range(0, len(Columns)):
        NameOfFeature = SV.nameofmeter(Data, i)  # get the name of the meter in column i
        if NameOfFeature != DT_Setup_object.NameOfSignal:  # making sure this method does not produce OwnLags
            Xauto = Data_AllSamples[NameOfFeature]  # copy of the Feature to use for shifting
            Score_best = (-100)  # set initial very bad value
            for lag in range(DT_Setup_object.MinFeatureLag, (DT_Setup_object.MaxFeatureLag + 1)):
                Score_1 = Score_best
                FeatureLagName = NameOfFeature + "_lag" + str(lag)  # create a column name per lag
                DataShift = Xauto.shift(periods=lag).fillna(
                    method='bfill')  # shift with the respective values with the respective lag
                Data_i = pd.concat([Data_c, DataShift.rename(FeatureLagName)], axis=1,
                                   join="inner")  # joining the dataframes just for the selected period
                # wrapper gets accuracy with respective feature lag
                (X_i, Y) = DT_Setup_object.split_signal_and_features(Data_i)
                X_train_i, X_test_i, y_train, y_test = train_test_split(X_i, Y, test_size=0.25)
                Result_dic = DT_Setup_object.EstimatorWrapper(X_train_i, y_train, X_test_i, y_test, *DT_Setup_object.WrapperParams)  # score will be done over hold out 0.25 percent of data
                Score_2 = Result_dic["score"]  # get the score with the additional featurelag
                # check whether score is higher than previous or not
                if Score_2 > Score_1:
                    DataShift_best = DataShift
                    FeatureLagName_best = FeatureLagName
                    Score_best = Score_2
            if Score_best > (
                    Score + DT_Setup_object.MinIncrease):  # if best featurelag of respective feature is better than initial score: add it
                Data = pd.concat([Data, DataShift_best.rename(FeatureLagName_best)], axis=1,
                                 join="inner")  # add best lag of respective feature to the dataframe
                print("Added feature lag = %s" % (FeatureLagName_best))
    return (Data)


# Main
def main(DT_Setup_object, DT_RR_object):
    print("FeatureConstruction")
    startTime = time.time()
    Data = pd.read_pickle(os.path.join(DT_Setup_object.PathToPickles, "ThePickle_from_PeriodSelection" + '.pickle'))
    Data_AllSamples = pd.read_pickle(os.path.join(DT_Setup_object.PathToPickles, "ThePickle_from_Preprocessing" + '.pickle'))

    Datas = [Data]  # also for not making e.g. featurelagcreate create lags of differences; Data needs to be in for the case no feature construction is done
    if DT_Setup_object.Cross_auto_cloud_correlation_plotting == True:
        cross_auto_cloud_correlation_plotting(DT_Setup_object, Data)
    if DT_Setup_object.DifferenceCreate == True:
        _Data = difference_create(DT_Setup_object, Data)
        Datas.append(_Data.drop(list(Data), axis=1))  # make sure only the added features are appended to Datas
    if DT_Setup_object.ManFeaturelagCreate == True:
        _Data = manual_featurelag_create(DT_Setup_object, Data, Data_AllSamples)
        Datas.append(_Data.drop(list(Data), axis=1))
    if DT_Setup_object.AutoFeaturelagCreate == True:
        _Data = auto_featurelag_constructor(DT_Setup_object, Data, Data_AllSamples)
        Datas.append(_Data.drop(list(Data), axis=1))
    if DT_Setup_object.ManOwnlagCreate == True:
        _Data = man_ownlag_create(DT_Setup_object, Data, Data_AllSamples)
        Datas.append(_Data.drop(list(Data), axis=1))

    DataF = pd.concat(Datas, axis=1, join="inner")  # joining the datas produced by all feature construction methods
    endTime = time.time()
    DT_RR_object.feature_construction_time = endTime - startTime
    DT_RR_object.df_feature_construction_data = DataF
    # Save dataframe to pickle
    DataF.to_pickle(os.path.join(DT_Setup_object.PathToPickles, "ThePickle_from_FeatureConstruction" + '.pickle'))

    # save dataframe in the ProcessedInputData excel file
    ExcelFile = os.path.join(DT_Setup_object.ResultsFolder, "ProcessedInputData_%s.xlsx" % (DT_Setup_object.NameOfExperiment))
    book = load_workbook(ExcelFile)
    writer = pd.ExcelWriter(ExcelFile, engine="openpyxl")
    writer.book = book
    writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
    DataF.to_excel(writer, sheet_name="FeatureConstruction")
    writer.save()
    writer.close()

import time
import sys
import os

import pandas as pd

from openpyxl import load_workbook
from sklearn.model_selection import train_test_split
import SharedVariablesFunctions as SVF
from core.data_tuning_optimizer.config.data_tuning_config import DataTuningSetup


# Todo: get the resolution of the data

# Cross, Cloud and Autocorrelation plots
# def cross_auto_cloud_correlation_plotting(DT_Setup_object, Data):
#     # set folder for sensor and resolution
#     CorrelationResultsFolder = "%s/%s" % (
#         DT_Setup_object.ResultsFolder,
#         "cross_auto_cloud_plotting",
#     )
#     # check if directory exists, if not, create it
#     if not os.path.exists(CorrelationResultsFolder):
#         os.makedirs(CorrelationResultsFolder)
#
#     # plot simple dependency on influence factors
#     Labels = [column for column in Data.columns]
#     for i in range(1, len(Labels)):
#         plot_x_y(
#             Data[Labels[i]],
#             Data[DT_Setup_object.NameOfSignal],
#             CorrelationResultsFolder,
#         )
#
#     # plot autocorrelation
#     plot_acf(
#         Data[DT_Setup_object.NameOfSignal],
#         CorrelationResultsFolder,
#         lags=DT_Setup_object.LagsToBePlotted,
#     )
#
#     # make an array for correlation coefficients
#     corrCoeffMax = np.zeros(len(Labels))
#     DictCoerrCoeff = dict()
#
#     # plot cross correlation between signal to be forecasted and exogenous input
#     for i in range(1, len(Labels)):
#         plot_crosscorr(
#             Data[DT_Setup_object.NameOfSignal],
#             Data[Labels[i]],
#             CorrelationResultsFolder,
#             lags=DT_Setup_object.LagsToBePlotted,
#         )
#         corrCoeff = plt.xcorr(
#             Data[DT_Setup_object.NameOfSignal],
#             Data[Labels[i]],
#             maxlags=DT_Setup_object.LagsToBePlotted,
#             normed=True,
#         )
#         # max value over the values of the correlation coefficient
#         DictCoerrCoeff[Data[Labels[i]].name] = np.amax(corrCoeff[1])
#         corrCoeffDf = pd.DataFrame(
#             DictCoerrCoeff,
#             index=[Data[DT_Setup_object.NameOfSignal].name],
#             columns=Labels,
#         )
#
#     # save cross correlations in an excel file
#     ExcelFile = "%s/CrossCorrelationCoefficients.xlsx" % (CorrelationResultsFolder)
#     writer = pd.ExcelWriter(ExcelFile)
#     corrCoeffDf.to_excel(
#         writer, sheet_name=DT_Setup_object.NameOfExperiment, float_format="%.2f"
#     )
#     writer.save()
#

# Manual creation of lags of the Features(Signal excluded)
def manual_featurelag_create(config, data, Data_AllSamples):
    if len(config["feature_lags"]) != len(list(data)):
        sys.exit(
            "Your FeatureLag Array has to have as many Arrays as Columns of your input data(Index excluded)"
        )
    for i in range(0, len(config.feature_lags)):
        NameOfFeature = SVF.nameofmeter(
            data, i
        )  # get the name of the meter in column i
        if (
            NameOfFeature != config.name_of_target
        ):  # making sure this method does not produce OwnLags
            Xauto = Data_AllSamples[
                NameOfFeature
            ]  # copy of the Feature to use for shifting
            for lag in range(0, len(config.feature_lags[i])):
                FeatureLagName = (
                    NameOfFeature + "_lag" + str(config.feature_lags[i][lag])
                )  # create a column name per lag
                DataShift = Xauto.shift(
                    periods=config.feature_lags[i][lag]
                ).fillna(
                    method="bfill"
                )  # shift with the respective values with the respective lag
                data = pd.concat(
                    [data, DataShift.rename(FeatureLagName)], axis=1, join="inner"
                )  # joining the dataframes just for the selected period
    return data


# Manual creation of lags of the Signal(OwnLags)
def man_ownlag_create(DT_Setup_object: DataTuningSetup, Data, Data_AllSamples):
    # is there a relevant autocorrelation for the signal to be predicted?
    Xauto = Data_AllSamples[
        DT_Setup_object.name_of_target
    ]  # copy of Y to use for shifting
    for lag in range(0, len(DT_Setup_object.target_lag)):
        OwnLagName = (
                DT_Setup_object.name_of_target + "_lag_" + str(DT_Setup_object.target_lag[lag])
        )  # create a column name per lag
        DataShift = Xauto.shift(periods=DT_Setup_object.target_lag[lag]).fillna(
            method="bfill"
        )  # shift with the lag to be considered
        Data = pd.concat(
            [Data, DataShift.rename(OwnLagName)], axis=1, join="inner"
        )  # joining the dataframes just for the selected period
    return Data


# Automatic creation of difference data through building the delta value of t - (t-1)
def difference_create(DT_Setup_object, Data):
    if (
        DT_Setup_object.create_differences == True
    ):  # for all features a derivative series should be constructed
        (X, Y) = SVF.split_signal_and_features(DT_Setup_object.name_of_target, Data)
    else:  # if certain features are selected
        (X, Y) = SVF.split_signal_and_features(DT_Setup_object.name_of_target, Data)
        X = X[
            X.columns[DT_Setup_object.create_differences]
        ]  # select only those columns of which a difference shall be created
    for column in X:  # loop through all columns
        DifferenceName = (
            "Delta_" + column
        )  # construct a new name for the difference data
        Difference = X[column].diff()  # build difference of respective feature
        Difference = Difference.fillna(
            0
        )  # fill up first row with 0 (since the first row has no value before and hence should be 0)
        Difference = SVF.post_scaler(
            Difference, DT_Setup_object.standard_scaling, DT_Setup_object.robust_scaling
        )
        Data = pd.concat(
            [Data, Difference.rename(columns={0: DifferenceName})], axis=1, join="inner"
        )  # joining the dataframes just for the selected period
    return Data


# automatic timeseries ownlag construction used in Feature Selection (Disadvantage: Only finds local optimum of the amount of Ownlags)
def automatic_timeseries_ownlag_constructor(DT_Setup_object, Data, Data_AllSamples):
    print("Auto timeseries-ownlag constructor START")
    Xauto = Data_AllSamples[
        DT_Setup_object.name_of_target
    ]  # copy of Y to use for shifting
    (X, Y) = SVF.split_signal_and_features(DT_Setup_object.name_of_target, Data)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)
    Result_dic = DT_Setup_object.wrapper_model(
        X_train, y_train, X_test, y_test, *DT_Setup_object.wrapper_params
    )  # score_test will be done over hold out 0.25 percent of data
    Score = Result_dic["score_test"]  # get the score_test  #get initial score_test
    Score_i = Score
    i = DT_Setup_object.minimum_target_lag - 1  # just for easier looping
    while True:  # loop as long as a new ownlag increases the accuracy
        i += 1
        Score = Score_i  # set score_test equal to the new and better score_i
        OwnLagName = (
                DT_Setup_object.name_of_target + "_lag_" + str(i)
        )  # create a column name per lag
        DataShift = Xauto.shift(periods=i).fillna(
            method="bfill"
        )  # shift with the lag to be considered
        Data = pd.concat(
            [Data, DataShift.rename(OwnLagName)], axis=1, join="inner"
        )  # joining the dataframes just for the selected period
        (X, Y) = SVF.split_signal_and_features(DT_Setup_object.name_of_target, Data)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)
        Result_dic = DT_Setup_object.wrapper_model(
            X_train, y_train, X_test, y_test, *DT_Setup_object.wrapper_params
        )  # score_test will be done over hold out 0.25 percent of data
        Score_i = Result_dic["score_test"]  # get the score_test with the additional ownlag
        if not (Score + DT_Setup_object.min_increase_4_wrapper) <= Score_i:
            break
    Data = Data.drop(
        [OwnLagName], axis=1
    )  # drop the last ownlag that was not improving the score_test
    return Data


def auto_featurelag_constructor(DT_Setup_object, Data, Data_AllSamples):
    print("auto featurelag constructor START")
    # wrapper gets default accuracy
    (X, Y) = SVF.split_signal_and_features(DT_Setup_object.name_of_target, Data)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)
    Result_dic = DT_Setup_object.wrapper_model(
        X_train, y_train, X_test, y_test, *DT_Setup_object.wrapper_params
    )  # score_test will be done over hold out 0.25 percent of data
    Score = Result_dic["score_test"]  # get the score_test  #get initial score_test
    Columns = list(Data)
    Data_c = Data  # copy of Data that is never changed
    for i in range(0, len(Columns)):
        NameOfFeature = SVF.nameofmeter(
            Data, i
        )  # get the name of the meter in column i
        if (
            NameOfFeature != DT_Setup_object.name_of_target
        ):  # making sure this method does not produce OwnLags
            Xauto = Data_AllSamples[
                NameOfFeature
            ]  # copy of the Feature to use for shifting
            Score_best = -100  # set initial very bad value
            for lag in range(
                DT_Setup_object.minimum_feature_lag, (DT_Setup_object.maximum_feature_lag + 1)
            ):
                Score_1 = Score_best
                FeatureLagName = (
                    NameOfFeature + "_lag" + str(lag)
                )  # create a column name per lag
                DataShift = Xauto.shift(periods=lag).fillna(
                    method="bfill"
                )  # shift with the respective values with the respective lag
                Data_i = pd.concat(
                    [Data_c, DataShift.rename(FeatureLagName)], axis=1, join="inner"
                )  # joining the dataframes just for the selected period
                # wrapper gets accuracy with respective feature lag
                (X_i, Y) = SVF.split_signal_and_features(
                    DT_Setup_object.name_of_target, Data_i
                )
                X_train_i, X_test_i, y_train, y_test = train_test_split(
                    X_i, Y, test_size=0.25
                )
                Result_dic = DT_Setup_object.wrapper_model(
                    X_train_i, y_train, X_test_i, y_test, *DT_Setup_object.wrapper_params
                )  # score_test will be done over hold out 0.25 percent of data
                Score_2 = Result_dic[
                    "score_test"
                ]  # get the score_test with the additional featurelag
                # check whether score_test is higher than previous or not
                if Score_2 > Score_1:
                    DataShift_best = DataShift
                    FeatureLagName_best = FeatureLagName
                    Score_best = Score_2
            if Score_best > (
                Score + DT_Setup_object.min_increase_4_wrapper
            ):  # if best featurelag of respective feature is better than initial score_test: add it
                Data = pd.concat(
                    [Data, DataShift_best.rename(FeatureLagName_best)],
                    axis=1,
                    join="inner",
                )  # add best lag of respective feature to the dataframe
                print("Added feature lag = %s" % (FeatureLagName_best))
    return Data


# Main
def main(DT_Setup_object, DT_RR_object):

    print("Feature Construction has begun...")
    startTime = time.time()

    Data = DT_RR_object.df_period_selection_data
    Data_AllSamples = DT_RR_object.df_preprocessing_data

    Datas = [
        Data
    ]  # also for not making e.g. featurelagcreate create lags of differences; Data needs to be in for the case no feature construction is done
    if DT_Setup_object.correlation_plotting == True:
        cross_auto_cloud_correlation_plotting(DT_Setup_object, Data)
    if DT_Setup_object.create_diff == True:
        _Data = difference_create(DT_Setup_object, Data)
        Datas.append(
            _Data.drop(list(Data), axis=1)
        )  # make sure only the added features are appended to Datas
    if DT_Setup_object.create_manual_feature_lags == True:
        _Data = manual_featurelag_create(DT_Setup_object, Data, Data_AllSamples)
        Datas.append(_Data.drop(list(Data), axis=1))
    if DT_Setup_object.create_automatic_feature_lags == True:
        _Data = auto_featurelag_constructor(DT_Setup_object, Data, Data_AllSamples)
        Datas.append(_Data.drop(list(Data), axis=1))
    if DT_Setup_object.create_manual_target_lag == True:
        _Data = man_ownlag_create(DT_Setup_object, Data, Data_AllSamples)
        Datas.append(_Data.drop(list(Data), axis=1))

    DataF = pd.concat(
        Datas, axis=1, join="inner"
    )  # joining the datas produced by all feature construction methods
    endTime = time.time()
    DT_RR_object.feature_construction_time = endTime - startTime
    DT_RR_object.df_feature_construction_data = DataF
    # Save dataframe to pickle
    DataF.to_pickle(
        os.path.join(
            DT_Setup_object.path_to_pickles,
            "ThePickle_from_FeatureConstruction" + ".pickle",
        )
    )

    # save dataframe in the ProcessedInputData excel file
    ExcelFile = os.path.join(
        DT_Setup_object.abs_path_to_result_folder,
        "ProcessedInputData_%s.xlsx" % (DT_Setup_object.name_of_tuning),
    )
    book = load_workbook(ExcelFile)
    writer = pd.ExcelWriter(ExcelFile, engine="openpyxl")
    writer.book = book
    writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
    DataF.to_excel(writer, sheet_name="FeatureConstruction")
    writer.save()
    writer.close()

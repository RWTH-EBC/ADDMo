import time

import pandas as pd

from openpyxl import load_workbook
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn_pandas import DataFrameMapper
import sys
from sklearn.externals import joblib
import numpy as np
import os

import SharedVariables as SV


##################################################################


##################################################################
# Manual Selection of Features (By the columns of the "FeatureConstruction" Excel Table)
def init_man_feature_select(DT_Setup_object, FeatureSelect, Data):
    if not DT_Setup_object.ColumnOfSignal in FeatureSelect:  # keep the column of signal (Check if the ColumnOfSignal is among the selected features)
        FeatureSelect = np.append(FeatureSelect,
                                  DT_Setup_object.ColumnOfSignal)  # add the column of signal to the features which shall be kept
    Data = Data[Data.columns[FeatureSelect]]  # select only those columns which shall be kept
    return (Data)


# Deal with NaN´s########################################################################################################
def NaNDealing(Data, NaNDealing):
    if NaNDealing == "ffill":
        Data = Data.ffill()  # fills NaN Values with the value before the NaN
    elif NaNDealing == "bfill":
        Data = Data.bfill()  # fills NaN Values with the value after NaN
    elif NaNDealing == "dropna":
        Data = Data.dropna()  # deletes every entire row(sample) with at least one NaN Value
    elif NaNDealing == "None":
        Data = Data
    else:
        print("Define a way how to deal NaN´s, if you don´t want to fill or delete NaN´s enter \"None\". "
              "Keep in mind that scaling and other procedures won´t work with NaN´s present")  # for GUI usage
        sys.exit("Define a way how to deal NaN´s, if you don´t want to fill or delete NaN´s enter \"None\". "
                 "Keep in mind that scaling and other procedures won´t work with NaN´s present")
    return (Data)


# Scaling################################################################################################################
def Scaling(DT_RR_object, StandardScaling, RobustScaling, NoScaling, Data):
    if sum(map(bool, [StandardScaling, RobustScaling, NoScaling])) != 1:  # Checking, that exactly one scaler is in use
        print("Please specify exactly 1 scaler for preprocessing!")  # for GUI usage
        sys.exit("Please specify exactly 1 scaler for preprocessing!")

    # Doing "StandardScaler"
    if StandardScaling == True:
        ScaleTracker_Signal = StandardScaler()
        ScaleTracker_Signal.fit(
            Data[SV.NameOfSignal].values.reshape(-1, 1))  # fit a scaler which is used to rescale afterwards
        DT_RR_object.scaler_obj = ScaleTracker_Signal  # adding runtime scaler object to DT Runtime Results object
        joblib.dump(ScaleTracker_Signal, os.path.join(SV.ResultsFolder,
                                                      "ScalerTracker.save"))  # dump this scaler in a file in the respective folder
        mapper = DataFrameMapper([(Data.columns, StandardScaler())])  # create the actually used scaler
        Scaled_Data = mapper.fit_transform(Data.copy())  # train it and scale the data
        Data = pd.DataFrame(Scaled_Data, index=Data.index, columns=Data.columns)
    # Doing "RobustScaler"
    if RobustScaling == True:
        ScaleTracker_Signal = RobustScaler()
        ScaleTracker_Signal.fit(
            Data[SV.NameOfSignal].values.reshape(-1, 1))  # fit a scaler which is used to rescale afterwards
        joblib.dump(ScaleTracker_Signal, os.path.join(SV.ResultsFolder,
                                                      "ScalerTracker.save"))  # dump this scaler in a file in the respective folder
        DT_RR_object.scaler_obj = ScaleTracker_Signal  # adding runtime scaler object to DT Runtime Results object
        mapper = DataFrameMapper([(Data.columns, RobustScaler())])  # create the actually used scaler
        Scaled_Data = mapper.fit_transform(Data.copy())  # train it and scale the data
        Data = pd.DataFrame(Scaled_Data, index=Data.index, columns=Data.columns)
    return (Data)

    # Resampling Procedure


def resample(Data, Resolution, WayOfResampling):
    if not len(list(Data)) == len(WayOfResampling):
        print("WayOfResampling array must have same amount of entries as columns of InputData")  # for GUI usage
        sys.exit("WayOfResampling array must have same amount of entries as columns of InputData")
    AggParameter = {list(Data)[0]: WayOfResampling[0]}  # create the dictionary for the .agg function
    for x in range(0, len(WayOfResampling)):
        AggParameter.update({list(Data)[x]: WayOfResampling[x]})  # add as many keys with resampling method as columns
    Data = Data.resample(Resolution).agg(AggParameter)  # Resamples data to a defined time frequence
    return Data

    # Todo: Rounding
    # Index = Data.index
    # Index = Index.round("15Min")
    # Data.set_index(
    #


######################################################################################

def main(DT_Setup_object, DT_RR_object):
    print("Preprocessing")
    startTime = time.time()

    # read dataframe from pickle
    Data = pd.read_pickle(os.path.join(SV.PathToPickles, "ThePickle_from_ImportData" + '.pickle'))

    # Execute functions if selected
    if True:
        Data = NaNDealing(Data, DT_Setup_object.NaNDealing)
    if DT_Setup_object.Resample == True:
        Data = resample(Data, DT_Setup_object.Resolution,
                        DT_Setup_object.WayOfResampling)  # Todo: review (Resample wurde von import data hier rüber kopiert, Resample was copied over here from import data)
    if DT_Setup_object.InitManFeatureSelect == True:
        Data = init_man_feature_select(DT_Setup_object, DT_Setup_object.InitFeatures, Data)
    if True:
        Data = Scaling(DT_RR_object, DT_Setup_object.StandardScaling, DT_Setup_object.RobustScaling,
                       DT_Setup_object.NoScaling, Data)

    endTime = time.time()
    DT_RR_object.preprocessing_time = endTime - startTime  # execution time
    DT_RR_object.df_preprocessing_data = Data
    # save dataframe to pickle
    Data.to_pickle(os.path.join(SV.PathToPickles, "ThePickle_from_Preprocessing" + '.pickle'))

    # save dataframe in the ProcessedInputData excel file
    ExcelFile = os.path.join(SV.ResultsFolder, "ProcessedInputData_%s.xlsx" % (DT_Setup_object.NameOfExperiment))
    book = load_workbook(ExcelFile)
    writer = pd.ExcelWriter(ExcelFile, engine="openpyxl")
    writer.book = book
    writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
    Data.to_excel(writer, sheet_name="Preprocessing")
    writer.save()
    writer.close()

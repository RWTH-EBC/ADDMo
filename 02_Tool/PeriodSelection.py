import time

import pandas as pd
from openpyxl import load_workbook
from Functions.PlotFcn import plot_TimeSeries
import os

# -------------------------------------------------------------------------------
def manual_period_select(Data, StartDate, EndDate):
    Data = Data[StartDate:EndDate]  # select given period (Manually selecting the period)
    print("Manual Period Selection")
    return Data


def timeseries_plotting(DT_Setup_object, Data, Scaled):
    # set folder for sensor and resolution
    CorrelationResultsFolder = "%s/%s" % (DT_Setup_object.ResultsFolder, "TimeSeries_plotting")
    # check if directory exists, if not, create it
    if not os.path.exists(CorrelationResultsFolder):
        os.makedirs(CorrelationResultsFolder)

    for column in list(Data):
        # plot time series
        plot_TimeSeries(df=Data[column], unitOfMeasure=column.split("[")[1].split("]")[0],
                        savePath=CorrelationResultsFolder, Scaled=Scaled, column=column)


# Main---------------------------------------------------------------------------
def main(DT_Setup_object, DT_RR_object):
    print("PeriodSelection")
    startTime = time.time()
    Data = pd.read_pickle(os.path.join(DT_Setup_object.PathToPickles, "ThePickle_from_Preprocessing" + '.pickle'))

    if DT_Setup_object.TimeSeriesPlot == True:
        if os.path.isfile(os.path.join(DT_Setup_object.ResultsFolder,
                                       "ScalerTracker.save")):  # check if a scaler is used, if a scaler is used the file "ScalerTracker" was created
            timeseries_plotting(Data, True)  # plot scaled data
            timeseries_plotting(pd.read_pickle(os.path.join(DT_Setup_object.PathToPickles, "ThePickle_from_ImportData" + '.pickle')),
                                False)  # plot raw data
        else:
            timeseries_plotting(DT_Setup_object, Data, False)
    if DT_Setup_object.ManSelect == True:
        Data = manual_period_select(Data, DT_Setup_object.StartDate, DT_Setup_object.EndDate)

    endTime = time.time()
    DT_RR_object.period_selection_time = endTime - startTime
    DT_RR_object.df_period_selection_data = Data
    # save dataframe to pickle
    Data.to_pickle(os.path.join(DT_Setup_object.PathToPickles, "ThePickle_from_PeriodSelection" + '.pickle'))

    # save dataframe in the ProcessedInputData excel file

    ExcelFile = os.path.join(DT_Setup_object.ResultsFolder, "ProcessedInputData_%s.xlsx" % (DT_Setup_object.NameOfExperiment))
    book = load_workbook(ExcelFile)
    writer = pd.ExcelWriter(ExcelFile, engine="openpyxl")
    writer.book = book
    writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
    Data.to_excel(writer, sheet_name="PeriodSelection")
    writer.save()
    writer.close()

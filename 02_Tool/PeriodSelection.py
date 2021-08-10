import time

import pandas as pd
from openpyxl import load_workbook
from Functions.PlotFcn import plot_TimeSeries
import os

# -------------------------------------------------------------------------------
def manual_period_select(DT_Setup_object, Data):
    Data = Data[DT_Setup_object.StartDate:DT_Setup_object.EndDate]  # select given period (Manually selecting the period)
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

    dataScaled = DT_Setup_object.df_preprocessing_data
    dataRaw = DT_Setup_object.df_import_data

    if DT_Setup_object.TimeSeriesPlot == True:
        if os.path.isfile(os.path.join(DT_Setup_object.ResultsFolder,"ScalerTracker.save")):  # check if a scaler is used, if a scaler is used the file "ScalerTracker" was created
            timeseries_plotting(DT_Setup_object, dataScaled, True)  # plot scaled data
            timeseries_plotting(DT_Setup_object, dataRaw, False)  # plot raw data
        else:
            timeseries_plotting(DT_Setup_object, dataScaled, False)
    if DT_Setup_object.ManSelect == True:
        dataScaled = manual_period_select(DT_Setup_object, dataScaled)

    endTime = time.time()
    DT_RR_object.period_selection_time = endTime - startTime
    DT_RR_object.df_period_selection_data = dataScaled
    # save dataframe to pickle
    dataScaled.to_pickle(os.path.join(DT_Setup_object.PathToPickles, "ThePickle_from_PeriodSelection" + '.pickle'))

    # save dataframe in the ProcessedInputData excel file

    ExcelFile = os.path.join(DT_Setup_object.ResultsFolder, "ProcessedInputData_%s.xlsx" % (DT_Setup_object.NameOfExperiment))
    book = load_workbook(ExcelFile)
    writer = pd.ExcelWriter(ExcelFile, engine="openpyxl")
    writer.book = book
    writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
    dataScaled.to_excel(writer, sheet_name="PeriodSelection")
    writer.save()
    writer.close()

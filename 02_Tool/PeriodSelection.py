import pandas as pd
from openpyxl import load_workbook
from Functions.PlotFcn import plot_TimeSeries
import sys
import os
import SharedVariables as SV




#-------------------------------------------------------------------------------
def manual_period_select(Data ,StartDate, EndDate):
    Data = Data[StartDate:EndDate] #select given period
    print("Manual Period Selection")
    return Data

def timeseries_plotting(Data, Scaled):
    # set folder for sensor and resolution
    CorrelationResultsFolder = "%s/%s" % (SV.ResultsFolder, "TimeSeries_plotting")
    # check if directory exists, if not, create it
    if not os.path.exists(CorrelationResultsFolder):
        os.makedirs(CorrelationResultsFolder)

    for column in list(Data):
        # plot time series
        plot_TimeSeries(df = Data[column], unitOfMeasure = column.split("[")[1].split("]")[0], savePath = CorrelationResultsFolder, Scaled=Scaled, column=column)

#Main---------------------------------------------------------------------------
def main():
    print("PeriodSelection")
    Data = pd.read_pickle(os.path.join(SV.PathToPickles, "ThePickle_from_Preprocessing" + '.pickle'))

    if SV.TimeSeriesPlot == True:
        if os.path.isfile(os.path.join(SV.ResultsFolder, "ScalerTracker.save")): #check if a scaler is used, if a scaler is used the file "ScalerTracker" was created
            timeseries_plotting(Data, True)  # plot scaled data
            timeseries_plotting(pd.read_pickle(os.path.join(SV.PathToPickles, "ThePickle_from_ImportData" + '.pickle')), False) #plot raw data
        else:
            timeseries_plotting(Data, False)
    if SV.ManSelect == True:
        Data = manual_period_select(Data, SV.StartDate, SV.EndDate)


    ##############################################
    #save dataframe to pickle
    Data.to_pickle(os.path.join(SV.PathToPickles, "ThePickle_from_PeriodSelection" + '.pickle'))

    # save dataframe in the ProcessedInputData excel file
    ExcelFile = os.path.join(SV.ResultsFolder, "ProcessedInputData_%s.xlsx"%(SV.NameOfExperiment))
    book = load_workbook(ExcelFile)
    writer = pd.ExcelWriter(ExcelFile, engine="openpyxl")
    writer.book = book
    writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
    Data.to_excel(writer, sheet_name="PeriodSelection")
    writer.save()
    writer.close()
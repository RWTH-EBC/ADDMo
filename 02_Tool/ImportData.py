import os
import time

import pandas as pd
import warnings

# Information about required Input shape -------------------------------------
# Input ExcelFile has to be named: "InputData" and saved in the Folder Data
# Sheet to read in must be the first sheet, with time as first column and all signals and features thereafter (one per column)
# The time must be in the format of "pandas.datetimeindex"
# Columns must have different names
# Each columns has to have a unit, which should be written like: [kwh] if no unit is available write []
# Sometimes the first row with the name of columns isn´t found as header; add a new header line, copy paste and delete the old one.
# -------------------------------------

# imports 1st Sheet ( in the ProcessedInputData_* ) as a dataframe and saves it as "ThePickle_from_ImportData"
def import_data(DT_Setup_object, DT_RR_object):
    Path = DT_Setup_object.InputData
    Data = pd.read_excel(io=Path, index_col=0)  # Column 0 has to be the Index Column; reads the excel file

    DT_RR_object.df_import_data = Data  # saving data to DataTuningRuntimeResults object
    Data.to_pickle(os.path.join(DT_Setup_object.PathToPickles, "ThePickle_from_ImportData" + '.pickle'))  # saves Data into a pickle

    # save dataframe in an excel file
    ExcelFile = os.path.join(DT_Setup_object.ResultsFolder, "ProcessedInputData_%s.xlsx" % DT_Setup_object.NameOfExperiment)
    writer = pd.ExcelWriter(ExcelFile)
    Data.to_excel(writer, sheet_name="ImportData")
    writer.save()
    writer.close()

    return Data

# main#######################################################################
def main(DT_Setup_object, DT_RR_object):
    print("ImportData")
    startTime = time.time()
    warnings.simplefilter(action="ignore", category=RuntimeWarning)
    print("Loading Input Data")
    import_data(DT_Setup_object, DT_RR_object)
    endTime = time.time()
    DT_RR_object.import_time = endTime - startTime

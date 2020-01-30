import os
import pandas as pd
import numpy as np
import datetime as dt
import sys
import warnings
#from GlobalVariables import *

import SharedVariables as SV


#Information about required Input shape -------------------------------------
#Input ExcelFile has to be named: "InputData" and saved in the Folder Data
#Sheet to read in must be the first sheet, with time as first column and all signals and features thereafter (one per column)
#The time must be in the format of "pandas.datetimeindex"
#Columns must have different names
#Each columns has to have a unit, which should be written like: [kwh] if no unit is available write []
#Sometimes the first row with the name of columns isn´t found as header; add a new header line, copy paste and delete the old one.
#-------------------------------------
#define clear (clean up) function to remove files that block the run
def clear():
    if os.path.isfile(os.path.join(SV.ResultsFolder, "ProcessedInputData_%s.xlsx"%(SV.NameOfExperiment))) == True: #check if path already exists
        Answer = input("Are you sure you want to delete or overwrite the data in %s: "%(SV.ResultsFolder))
        if Answer == "yes" or Answer == "Yes" or Answer == "y" or Answer == "Y":
            os.remove(os.path.join(SV.ResultsFolder, "ProcessedInputData_%s.xlsx"%(SV.NameOfExperiment))) #if it exists delete ProcessedInputData
            for FileName in os.listdir(SV.PathToPickles): #loop through all files in the directory and check if they are pickle files, if yes: delete them
                if FileName.endswith(".pickle"):
                    os.remove(os.path.join(SV.PathToPickles,FileName))
            if os.path.isfile(os.path.join(SV.ResultsFolder, "Settings_%s.xlsx"%(SV.NameOfExperiment))) == True: #check if there is a settings file, if yes delete it
                os.remove(os.path.join(SV.ResultsFolder, "Settings_%s.xlsx"%(SV.NameOfExperiment)))
            print("Files Deleted")

        else:
            sys.exit("Code stopped by user or invalid user input. Valid is Yes, yes, y and Y.") #stop the code

#imports 1st Sheet as a dataframe and saves it as "ThePickle"
def import_data():
    Path = SV.InputData
    Data = pd.read_excel(io=Path, index_col=0)                              #Column 0 has to be the Index Column; reads the excel file

    Data.to_pickle(os.path.join(SV.PathToPickles, "ThePickle_from_ImportData" + '.pickle'))  #saves Data into a pickle

    # save dataframe in an excel file
    ExcelFile = os.path.join(SV.ResultsFolder, "ProcessedInputData_%s.xlsx"%(SV.NameOfExperiment))
    writer = pd.ExcelWriter(ExcelFile)
    Data.to_excel(writer, sheet_name="ImportData")
    writer.save()
    writer.close()

    return Data

#main#######################################################################
def main():
    print("ImportData")
    warnings.simplefilter(action="ignore", category=RuntimeWarning)
    print("Loading Input Data")
    import_data()


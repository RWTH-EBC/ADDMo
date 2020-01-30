print("DataTuning")
#Package imports
import os
import pandas as pd
import time

#Pythonfiles Imports
import SharedVariables as SV
import ImportData
import Preprocessing
import PeriodSelection
import FeatureConstruction
import FeatureSelection

print("Module Import Section Done")

def main():
    #define path to data source files '.xls' & '.pickle'
    RootDir = os.path.dirname(os.path.realpath(__file__))
    PathToData = os.path.join(RootDir, 'Data')

    #Set Folder for Results
    ResultsFolder = os.path.join(RootDir, "Results", SV.NameOfData, SV.NameOfExperiment)
    PathToPickles = os.path.join(ResultsFolder, "Pickles")
    if not os.path.exists(ResultsFolder):
        os.makedirs(ResultsFolder)
        os.makedirs(PathToPickles)

    if SV.FixImport: #makes sure that the GUI can rename the directory and name of the inputdata if necessary(without Gui the data imported from the fixed place)
        InputData = os.path.join(PathToData, "InputData" + '.xlsx')
    else:
        InputData = os.path.join(PathToData, "GUI_Uploads", SV.GUI_Filename)

    #Set the found Variables in "SharedVariables"
    SV.RootDir = RootDir
    SV.PathToData = PathToData
    SV.ResultsFolder = ResultsFolder
    SV.PathToPickles = PathToPickles
    SV.InputData = InputData

    ImportData.clear() #make sure the selected folder is unused

    timestart = time.time()

    #Import the data
    ImportData.main()

    #Get the DataFrame produced by ImportData, this is a private variable
    __Data = pd.read_pickle(os.path.join(PathToPickles, "ThePickle_from_ImportData" + '.pickle'))
    NameOfSignal = list(__Data)[SV.ColumnOfSignal]
    SV.NameOfSignal = NameOfSignal #set Variable in "SharedVariables"

    #Preprocessing
    Preprocessing.main()

    #Period Selection
    PeriodSelection.main()

    #Feature Construction
    FeatureConstruction.main()

    #Feature selection
    FeatureSelection.main()

    timeend = time.time()

    #Documentation
    SV.documentation_DataTuning(timestart, timeend)

    print("Tuning the data took: %s seconds" %(timeend-timestart))
    print("End data tuning: %s/%s" % (SV.NameOfData, SV.NameOfExperiment))
    print("________________________________________________________________________\n")


if __name__ == "__main__":
    main()
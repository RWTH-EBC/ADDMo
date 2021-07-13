print("DataTuning")
# Package imports
import os
import pandas as pd
import time

# Pythonfiles Imports
import SharedVariables as SV
import ImportData
import Preprocessing
import PeriodSelection
import FeatureConstruction
import FeatureSelection
import DataTuningSetup as DTS
import DataTuningRuntimeResults as DTRR
from DataTuningRuntimeResults import DataTuningRuntimeResults

print("Module Import Section Done")  # imported all necessary files for data tuning
#create objects datasetup and datarunt

def main(DT_Setup_object):
    # define path to data source files '.xls' & '.pickle'
    RootDir = os.path.dirname(os.path.realpath(__file__))
    PathToData = os.path.join(RootDir, 'Data')

    # Set Folder for Results
    ResultsFolder = os.path.join(RootDir, "Results", SV.NameOfData, DT_Setup_object.NameOfExperiment)
    PathToPickles = os.path.join(ResultsFolder, "Pickles")
    if not os.path.exists(ResultsFolder):
        os.makedirs(ResultsFolder)
        os.makedirs(PathToPickles)

    if DT_Setup_object.FixImport:  # makes sure that the GUI can rename the directory and name of the inputdata if necessary(without Gui the data imported from the fixed place)
        InputData = os.path.join(PathToData, "InputData" + '.xlsx')
    else:
        InputData = os.path.join(PathToData, "GUI_Uploads", SV.GUI_Filename)

    # Set the found Variables in "SharedVariables"
    SV.RootDir = RootDir
    SV.PathToData = PathToData
    SV.ResultsFolder = ResultsFolder
    SV.PathToPickles = PathToPickles
    SV.InputData = InputData # todo: find out if it can be included in DT setup

    ImportData.clear()  # make sure the selected folder is unused

    DT_RR_object = DataTuningRuntimeResults()

    timestart = time.time()

    # Import the data
    ImportData.main(DT_Setup_object, DT_RR_object)

    # Get the DataFrame produced by ImportData, this is a private variable
    __Data = pd.read_pickle(os.path.join(PathToPickles, "ThePickle_from_ImportData" + '.pickle'))
    NameOfSignal = list(__Data)[SV.ColumnOfSignal]
    SV.NameOfSignal = NameOfSignal  # set Variable in "SharedVariables"

    # Preprocessing
    Preprocessing.main(DT_Setup_object, DT_RR_object)

    # Period Selection
    PeriodSelection.main(DT_Setup_object, DT_RR_object)

    # Feature Construction
    FeatureConstruction.main(DT_Setup_object, DT_RR_object)

    # Feature selection
    FeatureSelection.main(DT_Setup_object, DT_RR_object)

    timeend = time.time()

    # DataTuningSetup
    DTS.set_data(DT_Setup_object)
    DTRR.store_results(DT_RR_object)

    # Documentation
    #SV.documentation_DataTuning( endTime_FeatureSelection)

    print("Tuning the data took: %s seconds" % (timeend - timestart))
    print("End data tuning: %s/%s" % (SV.NameOfData, DT_Setup_object.NameOfExperiment))
    print("________________________________________________________________________\n")


if __name__ == "__main__":
    main()

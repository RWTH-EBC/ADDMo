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
from DataTuningRuntimeResults import DataTuningRuntimeResults as DTRR

print("Module Import Section Done")  # imported all necessary files for data tuning


# create objects datasetup and datarunt

def main(DT_Setup_object):
    # define path to data source files '.xls' & '.pickle'
    RootDir = os.path.dirname(os.path.realpath(__file__))
    PathToData = os.path.join(RootDir, 'Data')

    # Set Folder for Results
    ResultsFolder = os.path.join(RootDir, "Results", DT_Setup_object.NameOfData, DT_Setup_object.NameOfExperiment)
    PathToPickles = os.path.join(ResultsFolder, "Pickles")

    if not os.path.exists(ResultsFolder):
        os.makedirs(ResultsFolder)
        os.makedirs(PathToPickles)

    # makes sure that the GUI can rename the directory and name of the inputdata if necessary(without Gui the data imported from the fixed place)
    if DT_Setup_object.FixImport:
        InputData = os.path.join(PathToData, "InputData" + '.xlsx')
    else:
        InputData = os.path.join(PathToData, "GUI_Uploads", SV.GUI_Filename)

    # Save all the folder paths in the DTS object
    DT_Setup_object.RootDir = RootDir
    DT_Setup_object.PathToData = PathToData
    DT_Setup_object.ResultsFolder = ResultsFolder
    DT_Setup_object.PathToPickles = PathToPickles
    DT_Setup_object.InputData = InputData

    ImportData.clear(DT_Setup_object)  # make sure the selected folder is unused

    DT_RR_object = DTRR()  # create the DataTuningRuntimeResults object

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
    DT_Setup_object.dump_data()
    DT_RR_object.store_results()

    # Documentation
    # SV.documentation_DataTuning( endTime_FeatureSelection)

    print("Tuning the data took: %s seconds" % (timeend - timestart))
    print("End data tuning: %s/%s" % (DT_Setup_object.NameOfData, DT_Setup_object.NameOfExperiment))
    print("________________________________________________________________________\n")


if __name__ == "__main__":
    main()

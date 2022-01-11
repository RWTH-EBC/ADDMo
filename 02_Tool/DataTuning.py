'''
Executable to perform data tuning.
'''

import os
import time

import SharedVariables as SV
import ImportData
import Preprocessing
import PeriodSelection
import FeatureConstruction
import FeatureSelection
from DataTuningRuntimeResults import DataTuningRuntimeResults as DTRR

def main(DT_Setup_object):
    print("DataTuning")
    # define path to data source files '.xls' & '.pickle'
    RootDir = os.path.dirname(os.path.realpath(__file__))
    PathToData = os.path.join(RootDir, 'Data')

    # Set Folder for Results
    ResultsFolder = os.path.join(RootDir, "Results", DT_Setup_object.NameOfData, DT_Setup_object.NameOfExperiment) #Todo: could be directly set via setup class? (through getter?)
    PathToPickles = os.path.join(ResultsFolder, "Pickles")

    # makes sure that the GUI can rename the directory and name of the inputdata if necessary(without Gui the data imported from the fixed place)
    if DT_Setup_object.FixImport:
        path_input_data = os.path.join(PathToData, "InputData" + '.xlsx') #Todo: should be set in the setup class (either by default or by GUI) - delete all fiximport occurances - make GUI define the correct path
    else:
        path_input_data = os.path.join(PathToData, "GUI_Uploads", SV.GUI_Filename)

    # Save all the folder paths in the DTS object
    DT_Setup_object.RootDir = RootDir
    DT_Setup_object.PathToData = PathToData
    DT_Setup_object.ResultsFolder = ResultsFolder
    DT_Setup_object.PathToPickles = PathToPickles
    DT_Setup_object.InputData = path_input_data

    # create required folder structure for saving results
    SV.delete_and_create_folder(DT_Setup_object.ResultsFolder)
    os.makedirs(PathToPickles)

    DT_RR_object = DTRR()  # create the DataTuningRuntimeResults object

    timestart = time.time()

    # Import the data
    ImportData.main(DT_Setup_object, DT_RR_object)

    # Get the DataFrame produced by ImportData from DTRR object
    __Data = DT_RR_object.df_import_data
    NameOfSignal = list(__Data)[DT_Setup_object.ColumnOfSignal]
    DT_Setup_object.NameOfSignal = NameOfSignal  # save the variable in DTS object

    # Preprocessing
    Preprocessing.main(DT_Setup_object, DT_RR_object)

    # Period Selection
    PeriodSelection.main(DT_Setup_object, DT_RR_object)

    # Feature Construction
    FeatureConstruction.main(DT_Setup_object, DT_RR_object)

    # Feature selection
    FeatureSelection.main(DT_Setup_object, DT_RR_object)

    timeend = time.time()

    # Storing object data
    DT_Setup_object.dump_object()
    DT_RR_object.store_results(DT_Setup_object)

    # Documentation
    DT_Setup_object.documentation_DataTuning(timestart, timeend)

    print("Tuning the data took: %s seconds" % (timeend - timestart))
    print("End data tuning: %s/%s" % (DT_Setup_object.NameOfData, DT_Setup_object.NameOfExperiment))
    print("________________________________________________________________________\n")


if __name__ == "__main__":
    main()

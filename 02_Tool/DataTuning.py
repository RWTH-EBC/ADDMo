"""
Executable to perform data tuning.
"""

import os
import time

import SharedVariablesFunctions as SVF
import ImportData
import Preprocessing
import PeriodSelection
import FeatureConstruction
import FeatureSelection
from DataTuningRuntimeResults import DataTuningRuntimeResults as DTRR
from DataTuningSetup import DataTuningSetup as DTS
import Documentation as Document


def main(DT_Setup_object):
    print("Data Tuning process has begun...")

    if DT_Setup_object.PathToData == "Empty":
        DT_Setup_Object = SVF.setup_object_initializer(DT_Setup_object).dts

        # makes sure that the GUI can rename the directory and name of the inputdata if necessary(without Gui the data imported from the fixed place)
        if DT_Setup_object.FixImport:
            path_input_data = os.path.join(
                DT_Setup_Object.PathToData, "InputData" + ".xlsx"
            )  # Todo: should be set in the setup class (either by default or by GUI) - delete all fiximport occurances - make GUI define the correct path
        else:
            path_input_data = os.path.join(
                DT_Setup_Object.PathToData, "GUI_Uploads", SVF.GUI_Filename
            )

        DT_Setup_object.InputData = path_input_data

    # create required folder structure for saving results
    SVF.delete_and_create_folder(DT_Setup_object.ResultsFolder)
    os.makedirs(DT_Setup_object.PathToPickles)

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
    Document.documentation_DataTuning(DT_Setup_object, timestart, timeend)

    print("Tuning the data took: %s seconds" % (timeend - timestart))
    print(
        "End data tuning: %s/%s"
        % (DT_Setup_object.NameOfData, DT_Setup_object.NameOfExperiment)
    )
    print("________________________________________________________________________\n")


if __name__ == "__main__":
    DT_Setup_Object = DTS()
    DT_Setup_Object = SVF.setup_object_initializer(DT_Setup_Object).dts()
    SVF.GUI_Filename = input(
        "Please type the name of the input file (must be present in the GUI_Uploads folder):\n"
    )
    main(DT_Setup_Object)

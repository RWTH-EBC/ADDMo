import remi.gui as gui
import os
from remi import start, App
from BlackBoxes import (rf_predictor, RandomForestRegressor)
from sklearn.feature_selection import mutual_info_regression, f_regression
import DataTuning
import ModelTuning
import SharedVariables as SV
from SharedVariables import Hyperparametergrids, WrapperModels
import numpy as np
from multiprocessing import Process
from sklearn.model_selection import KFold, TimeSeriesSplit
print("Import in GUI_Remi.py done")


def convert_string_to_list(inputstring, operatordic, splitter = ","):
    List = []  # convert notation style into usable list of operators
    if operatordic is int or operatordic is float or operatordic is bool or operatordic is str: #convert string to int float or bool or string
        for i in inputstring.split(splitter):
            List.append(operatordic(i.strip()))
    else: #convert string to special operators defined in operatordic
        for i in inputstring.split(splitter):
            #operatordic = {"mean": np.mean, "sum": np.sum, "median": np.median}  # define the operators here
            List.append(operatordic[i.strip()])
    return List

def set_text(instance, text):
    instance.InfoFB.set_text(text)
    instance.InfoDT.set_text(text)
    instance.InfoMT.set_text(text)
    instance.InfoPO.set_text(text)

class container():
    def __init__(self, label, info_func):
        self.label = label
        self.info_func = info_func

    def create(self):
        self.container = gui.Widget(width='100%', layout_orientation=gui.Widget.LAYOUT_HORIZONTAL,
                                    margin='0px',
                                    style={'display': 'block', 'overflow': 'auto', "text-align": "left"})
        self.lbl = gui.Label(self.label, width="40%", height=20, margin="10px")
        self.lbl.set_on_click_listener(self.info_func)

    def merge(self, other):
        self.container.append([self.lbl, other])

class text(container):
    def __init__(self, label, info_func, text):
        super().__init__(label, info_func)
        self.text = text

    def do(self):
        super().create()
        txt = gui.TextInput(width="50%", height=20, margin='10px')
        txt.set_text(self.text)
        super().merge(txt)
        return self.container, txt

class spinbox(container):
    def __init__(self, label, info_func, initial, min, max):
        super().__init__(label, info_func)
        self.initial = initial
        self.min = min
        self.max = max

    def do(self):
        super().create()
        txt = gui.SpinBox(self.initial, self.min, self.max, width="50%", height=20, margin='10px')
        super().merge(txt)
        return self.container, txt

class checkbox(container):
    def __init__(self, label, info_func, initial, dialog = False):
        super().__init__(label, info_func)
        self.initial = initial
        self.dialog = dialog

    def do(self):
        super().create()
        txt = gui.CheckBox(False, width="50%", height=20, margin='10px')
        if self.dialog is not False:
            txt.set_on_change_listener(self.dialog)
        super().merge(txt)
        return self.container, txt

def onerowcheckbox(lbl1, default1,lbl2, default2):
            subcont = gui.Widget(width='100%', layout_orientation=gui.Widget.LAYOUT_HORIZONTAL,
                                                    margin='0px',
                                                    style={'display': 'block', 'overflow': 'auto'})
            checkbox1 = gui.CheckBoxLabel(" "+lbl1, default1, width="47%", height=20, margin='2px')
            checkbox2 = gui.CheckBoxLabel(" "+lbl2, default2, width="47%", height=20, margin='2px')
            subcont.append([checkbox1, checkbox2])
            return subcont, checkbox1, checkbox2

class dropdown(container):
    def __init__(self, label, info_func, entries):
        super().__init__(label, info_func)
        self.entries = entries

    def do(self):
        super().create()
        txt = gui.DropDown(width="50%", height=20, margin='10px')
        for key in self.entries:
            txt.append(self.entries[key], key)
        txt.select_by_key(next(iter(self.entries))) #select the first item of the entries dictionary as initial value
        super().merge(txt)
        return self.container, txt

class AutomatedTraining(App):
    def __init__(self, *args):
        super(AutomatedTraining, self).__init__(*args)

    def main(self):
        tb = gui.TabBox(width='100%', style={'display': 'block', 'overflow': 'hidden', "text-align": "center"})

        self.initialinfoboxtext = "Info will appear here upon clicking on the label in question. " \
                                  "Exemplary entries showing the required format can be found within the entrywidgets. "

        #FinalBayes tab
        if True:
            FinalBayesContainer = gui.Widget(width=500, margin='0px auto', style={'display': 'block', 'overflow': 'hidden', "text-align": "center"})

            self.Infolbl = gui.Label('Info Box', width="100%", height=30, margin='0px')
            self.InfoFB = gui.Label(self.initialinfoboxtext, width="100%", height="auto", margin='0px')


            #Creating subcontainer containing the entrywidget and a respective label
            subContainerNameOfData, self.NameOfDataFB = text("Name of data (results)", self.info_NameOfData, "TrialInput").do()

            subContainerNameOfSubTest = gui.Widget(width='100%', layout_orientation=gui.Widget.LAYOUT_HORIZONTAL,
                                                   margin='0px',
                                                   style={'display': 'block', 'overflow': 'auto',"text-align": "left"})
            self.lbl_NameOfSubTest = gui.Label("Name of subtest (results)", width="40%", height=20, margin="10px")
            self.txt_NameOfSubTest = gui.TextInput(width="50%", height=20, margin='10px')
            self.txt_NameOfSubTest.set_text('AutoML')
            self.lbl_NameOfSubTest.set_on_click_listener(self.info_NameOfSubTest)
            subContainerNameOfSubTest.append([self.lbl_NameOfSubTest, self.txt_NameOfSubTest])

            subContainerStartDateTraining = gui.Widget(width='100%', layout_orientation=gui.Widget.LAYOUT_HORIZONTAL,
                                                       margin='0px',
                                                       style={'display': 'block', 'overflow': 'auto',"text-align": "left"})
            self.lbl_StartDateTraining = gui.Label("Start of training", width="40%", height=20, margin="10px")
            self.txt_StartDateTraining = gui.TextInput(width="50%", height=20, margin='10px')
            self.txt_StartDateTraining.set_text('2016-08-01 00:00')
            self.lbl_StartDateTraining.set_on_click_listener(self.info_StartDateTraining)
            subContainerStartDateTraining.append([self.lbl_StartDateTraining, self.txt_StartDateTraining])

            subContainerEndDateTraining = gui.Widget(width='100%', layout_orientation=gui.Widget.LAYOUT_HORIZONTAL,
                                                     margin='0px',
                                                     style={'display': 'block', 'overflow': 'auto',"text-align": "left"})
            self.lbl_EndDateTraining = gui.Label("End of training", width="40%", height=20, margin="10px")
            self.txt_EndDateTraining = gui.TextInput(width="50%", height=20, margin='10px')
            self.txt_EndDateTraining.set_text('2016-08-14 23:45')
            self.lbl_EndDateTraining.set_on_click_listener(self.info_EndDateTraining)
            subContainerEndDateTraining.append([self.lbl_EndDateTraining, self.txt_EndDateTraining])

            subContainerStartDateTesting = gui.Widget(width='100%', layout_orientation=gui.Widget.LAYOUT_HORIZONTAL,
                                                      margin='0px',
                                                      style={'display': 'block', 'overflow': 'auto',"text-align": "left"})
            self.lbl_StartDateTesting = gui.Label("Start of testing", width="40%", height=20, margin="10px")
            self.txt_StartDateTesting = gui.TextInput(width="50%", height=20, margin='10px')
            self.txt_StartDateTesting.set_text('2016-08-15 00:00')
            self.lbl_StartDateTesting.set_on_click_listener(self.info_StartDateTesting)
            subContainerStartDateTesting.append([self.lbl_StartDateTesting, self.txt_StartDateTesting])

            subContainerEndDateTesting = gui.Widget(width='100%', layout_orientation=gui.Widget.LAYOUT_HORIZONTAL,
                                                    margin='0px',
                                                    style={'display': 'block', 'overflow': 'auto',"text-align": "left"})
            self.lbl_EndDateTesting = gui.Label("End of testing", width="40%", height=20, margin="10px")
            self.txt_EndDateTesting = gui.TextInput(width="50%", height=20, margin='10px')
            self.txt_EndDateTesting.set_text('2016-08-16 23:45')
            self.lbl_EndDateTesting.set_on_click_listener(self.info_EndDateTesting)
            subContainerEndDateTesting.append([self.lbl_EndDateTesting, self.txt_EndDateTesting])

            subContainerOwnlags = gui.Widget(width="100%", margin='0px auto', style={'display': 'block', 'overflow': 'hidden', "text-align": "left"})
            self.lbl_Ownlags = gui.Label("Define which ownlags shall be considered", width="100%", height=20, margin="0px")
            self.lbl_Ownlags.set_on_click_listener(self.info_Ownlags)
            subsubContainerCheckboxes1 = gui.Widget(width='100%', layout_orientation=gui.Widget.LAYOUT_HORIZONTAL,
                                                    margin='0px',
                                                    style={'display': 'block', 'overflow': 'auto'})
            self.checkbox_NoOL = gui.CheckBoxLabel('No ownlag', True, width="40%", height=20, margin='0px')
            self.checkbox_InertiaOL = gui.CheckBoxLabel('Inertia ownlag', False, width="40%", height=20, margin='0px')
            subsubContainerCheckboxes1.append([self.checkbox_NoOL, self.checkbox_InertiaOL])
            subsubContainerCheckboxes2 = gui.Widget(width='100%', layout_orientation=gui.Widget.LAYOUT_HORIZONTAL,
                                                    margin='0px',
                                                    style={'display': 'block', 'overflow': 'auto'})
            self.checkbox_TSOL = gui.CheckBoxLabel('Time series ownlag', False, width="40%", height=20, margin='0px')


            self.checkbox_PeriodOL = gui.CheckBoxLabel('Period ownlag', False, width="40%", height=20, margin='0px')
            self.checkbox_PeriodOL.set_on_change_listener(self.exec_dialog_periodOL_FB)
            subsubContainerCheckboxes2.append([self.checkbox_TSOL, self.checkbox_PeriodOL])
            subContainerOwnlags.append([self.lbl_Ownlags, subsubContainerCheckboxes1, subsubContainerCheckboxes2])

            subContainerFeatureConstruction = gui.Widget(width="100%", margin='0px auto', style={'display': 'block', 'overflow': 'hidden', "text-align": "left"})
            self.lbl_FeatureConstruction = gui.Label("Define which feature construction methods shall be considered", width="100%", height=20, margin="0px")
            self.lbl_FeatureConstruction.set_on_click_listener(self.info_FeatureConstruction)
            subsubContainerCheckboxesFC = gui.Widget(width='100%', layout_orientation=gui.Widget.LAYOUT_HORIZONTAL,
                                                    margin='0px',
                                                    style={'display': 'block', 'overflow': 'auto'})
            self.checkbox_Difference = gui.CheckBoxLabel('Difference', False, width="40%", height=20, margin='0px')
            self.checkbox_FeatureLag = gui.CheckBoxLabel('Feature lags', False, width="40%", height=20, margin='0px')
            self.checkbox_FeatureLag.set_on_change_listener(self.exec_dialog_featurelag_FB)
            subsubContainerCheckboxesFC.append([self.checkbox_Difference, self.checkbox_FeatureLag])
            subContainerFeatureConstruction.append([self.lbl_FeatureConstruction, subsubContainerCheckboxesFC])

            subContainerFileUpload = gui.Widget(width='100%', layout_orientation=gui.Widget.LAYOUT_HORIZONTAL,
                                                    margin='0px',
                                                    style={'display': 'block', 'overflow': 'auto',"text-align": "left"})
            self.lbl_UploadFile = gui.Label("Upload the input data", width="40%", height=30, margin="10px")
            self.bt_UploadFile = gui.FileUploader(os.path.join(os.path.dirname(os.path.realpath(__file__)), "Data", "GUI_Uploads"), width="50%", height=30, margin='10px')
            self.bt_UploadFile.set_on_success_listener(self.fileupload_on_success)
            self.bt_UploadFile.set_on_failed_listener(self.fileupload_on_failed)
            self.lbl_UploadFile.set_on_click_listener(self.info_UploadFile)
            subContainerFileUpload.append([self.lbl_UploadFile, self.bt_UploadFile])

            subContainerColumnOfSignal = gui.Widget(width='100%', layout_orientation=gui.Widget.LAYOUT_HORIZONTAL,
                                                    margin='0px',
                                                    style={'display': 'block', 'overflow': 'auto', "text-align": "left"})
            self.lbl_ColumnOfSignal = gui.Label("Column of signal", width="40%", height=20, margin="10px")
            self.txt_ColumnOfSignal = gui.SpinBox(0, 0, 1000, width="50%", height=20, margin='10px')
            self.lbl_ColumnOfSignal.set_on_click_listener(self.info_ColumnOfSignal)
            subContainerColumnOfSignal.append([self.lbl_ColumnOfSignal, self.txt_ColumnOfSignal])

            subContainerHyperBayesEval = gui.Widget(width='100%', layout_orientation=gui.Widget.LAYOUT_HORIZONTAL,
                                                    margin='0px',
                                                    style={'display': 'block', 'overflow': 'auto', "text-align": "left"})
            self.lbl_HyperBayesEval = gui.Label("Bayes eval hyperparameter", width="40%", height=20,
                                                margin="10px")
            self.txt_HyperBayesEval = gui.SpinBox(100, 1, 10000, width="50%", height=20, margin='10px')
            self.lbl_HyperBayesEval.set_on_click_listener(self.info_HyperBayesEval)
            subContainerHyperBayesEval.append([self.lbl_HyperBayesEval, self.txt_HyperBayesEval])

            subContainerFinalBayesEval = gui.Widget(width='100%', layout_orientation=gui.Widget.LAYOUT_HORIZONTAL,
                                                    margin='0px',
                                                    style={'display': 'block', 'overflow': 'auto', "text-align": "left"})
            self.lbl_FinalBayesEval = gui.Label("Bayes eval \"final bayes\"", width="40%", height=20, margin="10px")
            self.txt_FinalBayesEval = gui.SpinBox(50, 1, 10000, width="50%", height=20, margin='10px')
            self.lbl_FinalBayesEval.set_on_click_listener(self.info_FinalBayesEval)
            subContainerFinalBayesEval.append([self.lbl_FinalBayesEval, self.txt_FinalBayesEval])

            self.FB_Execute = gui.Button("Start computations \"Auto Final Bayes\"", width="100%", height=30, margin="0px")
            self.FB_Execute.set_on_click_listener(self.compute_FB)

            FinalBayesContainer.append([self.Infolbl, self.InfoFB])
            FinalBayesContainer.append([subContainerFileUpload, subContainerNameOfData, subContainerNameOfSubTest,
                                        subContainerStartDateTraining,
                                        subContainerEndDateTraining, subContainerStartDateTesting, subContainerEndDateTesting,
                                        subContainerOwnlags, subContainerFeatureConstruction,
                                        subContainerColumnOfSignal, subContainerHyperBayesEval, subContainerFinalBayesEval,
                                        self.FB_Execute])

            tb.add_tab(FinalBayesContainer, 'Auto Final Bayes', None)

        #Data tuning tab
        if True:
            DataTuningContainer = gui.Widget(width=500, margin='0px auto',
                                             style={'display': 'block', 'overflow': 'hidden', "text-align": "center"})

            self.InfoDT = gui.Label(self.initialinfoboxtext, width="100%", height="auto", margin='0px')

            ContDT_NameOfData, self.NameOfDataDT = text("Name of data (results)", self.info_NameOfData, "TrialInput").do()
            ContDT_NameOfExperiment, self.NameOfExperimentDT = text("Name of experiment (results)", self.info_NameOfExperiment, "TrialTunedData").do()
            ContDT_ColumnOfSignal, self.ColumnOfSignalDT = spinbox("Column of signal", self.info_ColumnOfSignal, 0,0,float("inf")).do()

            ContDT_FileUpload = gui.Widget(width='100%', layout_orientation=gui.Widget.LAYOUT_HORIZONTAL,
                                                    margin='0px',
                                                    style={'display': 'block', 'overflow': 'auto',"text-align": "left"})
            self.lbl_UploadFileDT = gui.Label("Upload the input data", width="40%", height=30, margin="10px")
            self.bt_UploadFileDT = gui.FileUploader('./Data/GUI_Uploads/', width="50%", height=30, margin='10px')
            self.bt_UploadFileDT.set_on_success_listener(self.fileupload_on_success)
            self.bt_UploadFileDT.set_on_failed_listener(self.fileupload_on_failed)
            self.lbl_UploadFileDT.set_on_click_listener(self.info_UploadFile)
            ContDT_FileUpload.append([self.lbl_UploadFileDT, self.bt_UploadFileDT])

            ContDT_Preprocessing = gui.Widget(width='100%', layout_orientation=gui.Widget.LAYOUT_HORIZONTAL,
                                                    margin='0px',
                                                    style={'display': 'block', 'overflow': 'auto',"text-align": "left"})
            lbl_preprocessing = gui.Label("Preprocessing", width="90%", height=20, margin="10px", style={"text-align": "center"})

            entries = {"bfill":"bfill", "ffill":"ffill", "dropna":"dropna"}
            ContDT_NanDealing, self.NaNDealing = dropdown("NaN dealing", self.info_NaNDealing, entries).do()
            ContDT_Resample, self.ResampleDT = checkbox("Resample the data`s resolution", self.info_resample, False, self.exec_dialog_Resample).do()
            ContDT_InitManFeatureSelectDT, self.InitManFeatureSelectDT = checkbox("Manual feature pre-selection", self.info_InitManFeatureSelect, False, self.exec_dialog_InitManFeatureSelect).do()
            entries = {"Standard":"Standard scaler", "Robust":"Robust scaler", "No":"Without scaling"}
            ContDT_Scaler, self.Scaler = dropdown("Scaler", self.info_Scaler, entries).do()
            ContDT_Preprocessing.append([lbl_preprocessing, ContDT_NanDealing, ContDT_Resample,
                                         ContDT_InitManFeatureSelectDT, ContDT_Scaler])

            ContDT_PeriodSelection = gui.Widget(width='100%', layout_orientation=gui.Widget.LAYOUT_HORIZONTAL,
                                                    margin='0px',
                                                    style={'display': 'block', 'overflow': 'auto',"text-align": "left"})
            lbl_PeriodSelection = gui.Label("Period selection", width="90%", height=20, margin="10px", style={"text-align": "center"})
            ContDT_TimeSeriesPlot, self.TimeSeriesPlotDT = checkbox("Time series plotting", self.info_TimeSeriesPlot, False).do()
            ContDT_ManPeriodSelection, self.ManPeriodSelectionDT = checkbox("Manual period selection", self.info_ManPeriodSelect,False, self.exec_dialog_ManPeriodSelectionDT).do()
            ContDT_PeriodSelection.append([lbl_PeriodSelection, ContDT_TimeSeriesPlot, ContDT_ManPeriodSelection])

            ContDT_FeatureConstruction = gui.Widget(width='100%', layout_orientation=gui.Widget.LAYOUT_HORIZONTAL,
                                                    margin='0px',
                                                    style={'display': 'block', 'overflow': 'auto',"text-align": "left"})
            lbl_FeatureConstruction = gui.Label("Feature Construction", width="90%", height=20, margin="10px",
                                            style={"text-align": "center"})
            ContDT_CrossAutoPlot, self.CrossAutoPlotDT = checkbox("Plot cross- & autocorrelation", self.info_CrossAutoPlot, False, self.exec_dialog_CrossAutoPlot).do()
            ContDT_Difference, self.DifferenceDT = checkbox("Difference", self.info_Difference, False, self.exec_dialog_DifferenceDT).do()
            ContDT_ManOwnLag, self.ManOwnLagDT = checkbox("Manual ownlag construct", self.info_ManOwnLag, False, self.exec_dialog_ManOwnLagDT).do()
            ContDT_AutoOwnLag, self.AutoOwnLagDT = checkbox("Auto time series ownlag", self.info_AutoOwnLag, False).do()
            ContDT_ManFeatureLag, self.ManFeatureLagDT = checkbox("Manual feature lag construct", self.info_ManFeatureLag, False, self.exec_dialog_ManFeatureLagDT).do()
            ContDT_AutoFeatureLag, self.AutoFeatureLagDT = checkbox("Auto feature lag construct", self.info_AutoFeatureLag, False, self.exec_dialog_AutoFeatureLagDT).do()


            ContDT_FeatureConstruction.append([lbl_FeatureConstruction, ContDT_CrossAutoPlot, ContDT_Difference,
                                               ContDT_ManOwnLag, ContDT_AutoOwnLag, ContDT_ManFeatureLag,
                                               ContDT_AutoFeatureLag])

            ContDT_FeatureSelection = gui.Widget(width='100%', layout_orientation=gui.Widget.LAYOUT_HORIZONTAL,
                                                    margin='0px',
                                                    style={'display': 'block', 'overflow': 'auto',
                                                           "text-align": "left"})
            lbl_FeatureSelection = gui.Label("Feature Selection", width="90%", height=20, margin="10px",
                                                style={"text-align": "center"})
            ContDT_ManFeatureSelect, self.ManFeatureSelectDT = checkbox("Manual feature selection",
                                                                  self.info_ManFeatureSelect, False,
                                                                  self.exec_dialog_ManFeatureSelectDT).do()
            ContDT_LowVarFilter, self.LowVarFilterDT = checkbox("Low variance filter", self.info_LowVarFilter, False,
                                                            self.exec_dialog_LowVarFilterDT).do()
            ContDT_ICA, self.ICADT = checkbox("ICA", self.info_ICADT, False).do()
            ContDT_UnivariateFilter, self.UnivariateFilterDT = checkbox("Univariate filter",
                                                                  self.info_UnivariateFilterDT, False,
                                                                  self.exec_dialog_UnivariateFilterDT).do()
            ContDT_UnivariateEmbedded, self.UnivariateEmbeddedDT = checkbox("Univariate embedded",
                                                                    self.info_UnivariateEmbedded, False,
                                                                    self.exec_dialog_UnivariateEmbeddedDT).do()
            ContDT_MultivariateEmbedded, self.MultivariateEmbeddedDT = checkbox("Multivariate embedded",
                                                                    self.info_MultivariateEmbedded, False,
                                                                    self.exec_dialog_MultivariateEmbeddedDT).do()
            ContDT_WrapperRecursive, self.WrapperRecursiveDT = checkbox("Wrapper selection",
                                                                    self.info_WrapperRecursiveDT, False, False).do()

            ContDT_FeatureSelection.append([lbl_FeatureSelection, ContDT_ManFeatureSelect, ContDT_LowVarFilter,
                                            ContDT_ICA, ContDT_UnivariateFilter, ContDT_UnivariateEmbedded,
                                            ContDT_MultivariateEmbedded, ContDT_WrapperRecursive])

            self.DT_Execute = gui.Button("Start computations \"Data tuning\"", width="100%", height=30, margin="0px")
            self.DT_Execute.set_on_click_listener(self.compute_DT)

            DataTuningContainer.append([self.Infolbl, self.InfoDT])
            DataTuningContainer.append([ContDT_NameOfData, ContDT_NameOfExperiment, ContDT_ColumnOfSignal,
                                        ContDT_FileUpload, ContDT_Preprocessing, ContDT_PeriodSelection,
                                        ContDT_FeatureConstruction, ContDT_FeatureSelection, self.DT_Execute])

            tb.add_tab(DataTuningContainer, 'Data tuning', None)

        #ModelTuning tab
        if True:
            ModelTuningContainer = gui.Widget(width=500, margin='0px auto',
                                             style={'display': 'block', 'overflow': 'hidden', "text-align": "center"})

            self.InfoMT = gui.Label(self.initialinfoboxtext,
                width="100%", height="auto", margin='0px')

            ContMT_NameOfData, self.NameOfDataMT = text("Name of data (as input)", self.info_NameOfData, "TrialInput").do()
            ContMT_NameOfExperiment, self.NameOfExperimentMT = text("Name of experiment (as input)", self.info_NameOfExperiment, "TrialTunedData").do()
            ContMT_NameOfSubTest, self.NameOfSubTestMT = text("Name of subtest (results)", self.info_NameOfSubTest, "TrialTunedModel").do()
            ContMT_StartTraining, self.StartTrainingMT = text("Start of training", self.info_StartDateTraining, '2016-08-01 00:00').do()
            ContMT_EndTraining, self.EndTrainingMT = text("End of training", self.info_EndDateTraining, '2016-08-14 23:45').do()
            ContMT_StartTesting, self.StartTestingMT = text("Start of testing", self.info_StartDateTesting, '2016-08-15 00:00').do()
            ContMT_EndTesting, self.EndTestingMT = text("End of testing", self.info_EndDateTesting, '2016-08-16 23:45').do()
            ContMT_HyperBayesEval, self.HyperBayesEvalMT = spinbox("Bayes eval hyperparameter", self.info_HyperBayesEval, 100, 1, 10000).do()

            ContMT_CV = gui.Widget(width='100%', layout_orientation=gui.Widget.LAYOUT_HORIZONTAL,
                                                    margin='0px',
                                                    style={'display': 'block', 'overflow': 'auto',
                                                           "text-align": "left"})
            lbl = gui.Label("Cross-validation", width="40%", height=20, margin="10px")
            lbl.set_on_click_listener(self.info_CV)
            self.CVTypeMT = gui.DropDown(width="40%", height=20, margin='10px')
            entries = {"KFold":"KFold", "TimeSeriesSplit":"TimeSeriesSplit"}
            for key in entries:
                self.CVTypeMT.append(entries[key], key)
            self.CVTypeMT.select_by_key(next(iter(entries)))  # select the first item of the entries dictionary as initial value
            self.CVFoldsMT = gui.SpinBox(3,1,100, width="6%", height=20, margin="10px")
            ContMT_CV.append([lbl, self.CVTypeMT, self.CVFoldsMT])

            ContMT_Shuffle, self.ShuffleMT = checkbox("Shuffle", self.info_shuffle, False).do()
            ContMT_Recursive, self.RecursiveMT = checkbox("Recursive prediction", self.info_Recursive, False).do()

            ContMT_Model = gui.Widget(width="100%", margin='0px auto',
                                             style={'display': 'block', 'overflow': 'hidden', "text-align": "left"})
            self.lbl_ModelMT = gui.Label("Define which models shall be used", width="100%", height=20, margin="0px")
            self.lbl_ModelMT.set_on_click_listener(self.info_ModelMT)
            subcont1, self.MT_RF, self.MT_SVR = onerowcheckbox("Random forest", False, "SVR (BayesOpt)", False)
            subcont2, self.MT_ANN, self.MT_GB = onerowcheckbox("ANN (BayesOpt)", False, "GradientBoost (BayesOpt)", False)
            subcont3, self.MT_Lasso, self.MT_Modelselect = onerowcheckbox("Lasso (BayesOpt)", False, "ModelSelect", False)
            subcont4, self.MT_SVR_grid, self.MT_ANN_grid = onerowcheckbox("SVR (grid search)", False, "ANN (grid search)", False)
            subcont5, self.MT_GB_grid, self.MT_Lasso_grid = onerowcheckbox("GradientBoost (grid search)", False, "Lasso",False)
            ContMT_Model.append([self.lbl_ModelMT, subcont1, subcont2, subcont3, subcont4, subcont5])

            ContMT_IndivModel, self.IndivModelMT = dropdown("Type of \"individual model\"", self.info_IndivModelMT, {"No":"No individual model", "week_weekend":"Weekday/weekend model", "hourly":"One model per hour (24models)", "byFeature":"By feature threshold"}).do()
            self.IndivModelMT.set_on_change_listener(self.exec_dialog_indivmodelbyfeature_MT)

            self.mt_Execute = gui.Button("Start computations \"Model tuning\"", width="100%", height=30, margin="0px")
            self.mt_Execute.set_on_click_listener(self.compute_MT)

            ModelTuningContainer.append([self.Infolbl, self.InfoMT])
            ModelTuningContainer.append([ContMT_NameOfData, ContMT_NameOfExperiment, ContMT_NameOfSubTest,
                                         ContMT_StartTraining, ContMT_EndTraining, ContMT_StartTesting,
                                         ContMT_EndTesting, ContMT_HyperBayesEval, ContMT_CV, ContMT_Recursive,
                                         ContMT_Shuffle, ContMT_IndivModel, ContMT_Model,
                                         self.mt_Execute])

            tb.add_tab(ModelTuningContainer, 'Model tuning', None)

        #Predict only tab
        if True:
            PredictOnlyContainer = gui.Widget(width=500, margin='0px auto',
                                             style={'display': 'block', 'overflow': 'hidden', "text-align": "center"})

            self.InfoPO = gui.Label(self.initialinfoboxtext,
                width="100%", height="auto", margin='0px')

            ContPO_NameOfData, self.NameOfDataPO = text("Name of data (as input)", self.info_NameOfData, "TrialInput").do()
            ContPO_NameOfExperiment, self.NameOfExperimentPO = text("Name of experiment (as input)", self.info_NameOfExperiment,
                                                                    "TrialTunedData").do()
            ContPO_NameOfSubTest, self.NameOfSubTestPO = text("Name of subtest (as input)", self.info_NameOfSubTest, "TrialTunedModel").do()
            ContPO_NameOfOnlyPredict, self.NameOfOnlyPredictPO = text("Name of prediction (results)", self.info_NameOfOnlyPredict, "TrialOnlyPredict").do()
            ContPO_StartTesting, self.StartTestingPO = text("Start of testing", self.info_StartDateTesting,
                                                            '2016-08-01 00:00').do()
            ContPO_EndTesting, self.EndTestingPO = text("End of testing", self.info_EndDateTesting, '2016-08-16 23:45').do()

            ContPO_Recursive, self.RecursivePO = checkbox("Recursive prediction", self.info_Recursive, False, False).do()
            ContPO_ValidationPeriod, self.ValidationPeriodPO = checkbox("Custom validation period", self.info_ValidationPeriodPO, False, self.exec_dialog_validationperiod_PO).do()



            self.PO_Execute = gui.Button("Start computations \"Predict only\"", width="100%", height=30, margin="0px")
            self.PO_Execute.set_on_click_listener(self.compute_PO)

            PredictOnlyContainer.append([self.Infolbl, self.InfoPO])
            PredictOnlyContainer.append([ContPO_NameOfData, ContPO_NameOfExperiment, ContPO_NameOfSubTest,
                                         ContPO_NameOfOnlyPredict, ContPO_StartTesting, ContPO_EndTesting, ContPO_Recursive,
                                         ContPO_ValidationPeriod, self.PO_Execute])

            tb.add_tab(PredictOnlyContainer, 'Predict only', None)

        return tb

    #Define the listener functions to update the infobox
    if True:
        def nothing(self, widget, value="optional"): #work around function to have some widget without infofunction
            pass

        def info_NameOfData(self, widget):
            text = 'Defines the name of the subfolder in which the results are saved. It is recommended to name it after the input data.'
            set_text(self, text)

        def info_NameOfSubTest(self, widget):
            text = 'Defines the name of the subfolder in which the specific results of "model tuning" or "final bayes" are saved. ' \
                   'It is a subfolder of the "name of experiment" folder. '
            set_text(self, text)

        def info_NameOfExperiment(self, widget):
            text = 'Name of experiment defines the name of the subfolder in which the results of "data tuning" are saved. ' \
                   'It is a subfolder of the "name of data" folder. '
            set_text(self, text)

        def info_NameOfOnlyPredict(self, widget):
            text = "Defines the name of the subfolder in which the results of only predict are saved. " \
                   'It is a subfolder of the "name of subtest" folder. '
            set_text(self, text)

        def info_StartDateTraining(self, widget):
            text = ('Date and time when the period for training the models shall start. Format: yyyy-mm-dd hh:mm. ')
            set_text(self, text)

        def info_EndDateTraining(self, widget):
            text = ('Date and time when the period for training the models shall end. Format: yyyy-mm-dd hh:mm. ')
            set_text(self, text)

        def info_StartDateTesting(self, widget):
            text = ('Date and time when the period for testing the models shall start. Format: yyyy-mm-dd hh:mm. ')
            set_text(self, text)

        def info_EndDateTesting(self, widget):
            text = ('Date and time when the period for testing the models shall end. Format: yyyy-mm-dd hh:mm. ')
            set_text(self, text)

        def info_Ownlags(self, widget):
            text = ('Defines which types of ownlags shall be considered. '
                                   'Each ticked box will create a distinctive tuned data set with the respective ownlags. '
                                   'Information on the types of ownlags and when they should be considered can be found in '
                                   '"Automated Data Driven Modeling of Building Energy Systems via Machine Learning Algorithms"'
                                   ', a master thesis by Martin Rätz (RWTH internal library). '
                    'We call lagged time series of either the signal or a feature “ownlag” and “featurelag” respectively. '
                    '''Inertia ownlag: One small ownlag to track the system’s inertia.
It can be physically valid. Use if the system has inertia
that is significant in the respective resolution. Expected
model behavior: Model starts of from the inertia ownlag
value and adds changes to that.
Period ownlag: One or several ownlags lagged by a typical
period, e.g. 24 hours or a week. It is not physically
valid. Use if unknown influences affect the system and
the system is following any kind of repeating periodic
pattern, e.g. a system affected by unknown occupancies
or control schedules. Expected model behavior: Model
copies pattern from, e.g. yesterday, and adds changes to
that.
Time series ownlag: A set of ownlags in a row, regularly
starting from the most recent lag (inertia ownlag). It is
regularly not physically valid, except the system depends
on the derivative of its inertia. Use if unknown influences
affect the system and the system is following any kind
of pattern; “native recurrent models” are recommended.
Expected model behavior: Model tracks typical patterns,
such as trend, periods, etc. Future values are guessed by
continuing the patterns and adding changes to that. 
                    ''')
            set_text(self, text)

        def info_FeatureConstruction(self, widget):
            text = ("""Difference creates the discrete "derivative" for all features. 
            Feature lag creator creates lagged series of features, the optimal lag is chosen automatically. 
            All created features are added to the existing feature space.""")
            set_text(self, text)

        def info_UploadFile(self, widget):
            text = """The Input Data has the following restrictions:
            1. Input must be an ExcelFile
            2. The data must be on the first excel sheet, with time as first column and all signals and features thereafter (one per column)
            3. The time must be in the format of "pandas.datetimeindex"
            4. Columns must have different names
            5. Each columns has to have a unit, which should be written after the name like: [kwh]. If no unit is available write []
            6. The index should be continuous (no missing steps)"""
            set_text(self, text)

        def fileupload_on_success(self, widget, filename):
            set_text(self, 'File upload success: ' + filename)
            SV.GUI_Filename = filename

        def fileupload_on_failed(self, widget, filename):
            set_text(self, 'File upload failed: ' + filename)

        def info_ColumnOfSignal(self, widget):
            text = 'Define the column of the input file in which the signal is. The signal is the variable that is to be ' \
                   'forecasted. Count up from 0 in the first column after the index (1st column after index = 0)'
            set_text(self, text)

        def info_HyperBayesEval(self, widget):
            text = ('Define the number of bayesian evaluations for tuning the hyperparameter of the models.'
                                   ' The more evaluations the more precise and time consuming')
            set_text(self, text)

        def info_FinalBayesEval(self, widget):
            text = ('Define the number of bayesian evaluations for the final bayesian optimization. '
                                   'Optimizing the type of model, the selected features and the type of individual model. '
                                   'The more evaluations the more precise and time consuming')
            set_text(self, text)

        def info_ValidationPeriodPO(self, widget):
            text = ('If ticked define the period on which the validation through the mean, standard deviation and max error shall be conducted. '
                    'Otherwise the whole period of the data is used for this. The period will be split into periods with the length of the prediction horizon length, which is set by the test period. ')
            set_text(self, text)

        def info_Recursive(self, widget):
            text = ("If enabled the prediction is done recusively. Only necessary if ownlags are used that are smaller than the prediction horizon. "
                    '''Recursive prediction iteratively passes the one step ahead
predicted signal to the respective ownlags of the future
samples. In that way the subsequent step can be predicted
with the ownlag discovered in the previous prediction step,
and so forth. A disadvantage is that the prediction error of
the first step is fed to the prediction of the next step, possibly
increasing the error with each step.''')
            set_text(self, text)

        def info_ModelMT(self, widget):
            text = ("Select a model. BayesOpt and grid search mark the used type of hyperparameter optimization for the respective model. Modelselection runs all BayesOpt models and determines the best")
            set_text(self, text)

        def info_IndivModelMT(self, widget):
            text = ("Type of individual model that is to be used. Own individual models can be defined in \"BlackBoxes.py\". "
                    """Individual model creates individual sample batches. In
model tuning, one individual model is trained per batch. An
illustrative example for individual model with a grouping
by time is dividing a data set into a weekdays batch and
a weekends batch. Accordingly, two individual models are
achieved, one trained on weekdays and another one trained
on weekends. While using the model, e.g. for forecasting, the
weekdays model is used only to forecast weekdays and the
weekends model only for the weekends.""")
            set_text(self, text)

        def info_shuffle(self, widget):
            text = ("Randomly shuffles the train data. This may increase accuracy through unbiased training.")
            set_text(self, text)

        def info_CV(self, widget):
            text = ("Cross-validation avoids overfitting while hyperparameter tuning. Select a type of cross-validation and the respective number of folds. "
                    "More information on \"Scikit-learn.org\" "
                    'Additional cross-validation types can be added in GUI_Remi.py')
            set_text(self, text)

        def info_resample(self, widget):
            text = ("Resamples the resolution of the input data.")
            set_text(self, text)

        def info_InitManFeatureSelect(self, widget):
            text = ("Manual selection of features by their column number")
            set_text(self, text)

        def info_NaNDealing(self, widget):
            text = ("Define how NaN´s shall be handled. Description of the operators can be found on scikit-learn.org.")
            set_text(self, text)

        def info_Scaler(self, widget):
            text = ("Define a scaler, information can be found on scikit-learn.org. Scaling is recommended in the very most cases.")
            set_text(self, text)

        def info_TimeSeriesPlot(self, widget):
            text = ("If true all features and signals are plotted and exported to the results folder. Can be used to determine faulty periods or to get an general idea of the data.")
            set_text(self, text)

        def info_ManPeriodSelect(self, widget):
            text = ("Select a period out of the overall data. In model tuning you can select train and test periods, so the period selection here is only meant to exclude periods or to reduce computation time of data tuning. ")
            set_text(self, text)

        def info_CrossAutoPlot(self, widget):
            text = ("Creates cross- & autocorrelation plots of all the features and signal. The autocorrelation plot can be used to determine significant ownlags, the crosscorrelation plot significant featurelags")
            set_text(self, text)

        def info_Difference(self, widget):
            text = ("Difference creates the discrete \"derivative\" of features.")
            set_text(self, text)

        def info_ManOwnLag(self, widget):
            text = ("Creates custom ownlags. "
                    "We call lagged time series of either the "
                    "signal or a feature “ownlag” and “featurelag” respectively.")
            set_text(self, text)

        def info_AutoOwnLag(self, widget):
            text = ("Creates time series ownlags automatically. Being a wrapper method it adds ownlags until the accuracy does not increase anymore."
                    """We call lagged time series of either the
signal or a feature “ownlag” and “featurelag” respectively. 
                    Time series ownlag: A set of ownlags in a row, regularly
starting from the most recent lag (inertia ownlag). It is
regularly not physically valid, except the system depends
on the derivative of its inertia. Use if unknown influences
affect the system and the system is following any kind
of pattern; “native recurrent models” are recommended.
Expected model behavior: Model tracks typical patterns,
such as trend, periods, etc. Future values are guessed by
continuing the patterns and adding changes to that.""")
            set_text(self, text)

        def info_ManFeatureLag(self, widget):
            text = ("Custom feature lags are constructed. "
                    """We call lagged time series of either the
signal or a feature “ownlag” and “featurelag” respectively. 
                    Featurelag: One or several lagged values of a feature. It can be
physically valid. Use if a feature has a delayed influence
on the signal. Expected model behavior: Model is able to
grasp the influence of the lagged feature on the signal,
e.g. lagged radiator temperature as feature to the room
temperature as signal. """)
            set_text(self, text)

        def info_AutoFeatureLag(self, widget):
            text = ("Creates lagged series of features, the optimal lag is chosen automatically via a wrapper method. "
                    """We call lagged time series of either the
signal or a feature “ownlag” and “featurelag” respectively. 
                    Featurelag: One or several lagged values of a feature. It can be
physically valid. Use if a feature has a delayed influence
on the signal. Expected model behavior: Model is able to
grasp the influence of the lagged feature on the signal,
e.g. lagged radiator temperature as feature to the room
temperature as signal. """)
            set_text(self, text)

        def info_ManFeatureSelect(self, widget):
            text = ("Manual selection of features by their column number")
            set_text(self, text)

        def info_LowVarFilter(self, widget):
            text = ("Delete features which time series have a low variance (variance value is calculated for the scaled data).")
            set_text(self, text)

        def info_ICADT(self, widget):
            text = ("Independent component analysis, see scikit-learn.org for more information.")
            set_text(self, text)

        def info_UnivariateFilterDT(self, widget):
            text = ("Several univariate filter methods. Information can be found in scikit-learn.org and in 'An Introduction to Variable and Feature Selection' from Isabelle Guyon which uses the term \"variable ranking\" for univariate")
            set_text(self, text)

        def info_UnivariateEmbedded(self, widget):
            text = ("Univariate embedded method. Information can be found on scikit-learn.org and in "
                    "'An Introduction to Variable and Feature Selection' from Isabelle Guyon "
                    "which uses the term \"variable ranking\" for univariate)")
            set_text(self, text)

        def info_MultivariateEmbedded(self, widget):
            text = ("Recursive embedded feature selection, a multivariate embedded method. Information can be found in "
                    ", scikit-learn.org and 'An Introduction to Variable and Feature Selection' from Isabelle Guyon "
                    "which uses the term \"variable subset selection\" for multivariate. ")
            set_text(self, text)

        def info_WrapperRecursiveDT(self, widget):
            text = ("Recursive wrapper feature selection. Deletes features as long as it improves the forecast accuracy. "
                    "Not recommended, information can be found in "
                    '"Automated Data Driven Modeling of Building Energy Systems via Machine Learning Algorithms"'
                    ', a master thesis by Martin Rätz (RWTH internal library)')
            set_text(self, text)

        #info fields within the dialogs
        def info_Date(self, widget):
            text = ("Only the selected data period will be processed. Entry format: yyyy-mm-dd hh:mm")
            self.InfoBoxManPeriodSelectionDT.set_text(text)

        def info_InitFeatures(self, widget):
            text = ("Enter the column number of the features you want to keep, separated by a comma. (Start to count from 0 with first Column after Index, Column of signal needs to be counted, but will be kept in any case)")
            self.InfoBoxInitManFeatureSelect.set_text(text)

        def info_Resolution(self, widget):
            text = ("Define the desired resolution. min for minutes, s for seconds")
            self.InfoBoxResample.set_text(text)

        def info_WayOfResampling(self, widget):
            text = ("Define the way how the data shall be transformed. This needs to be defined for each column, separated by a comma. Possible operators are: mean, sum and median")
            self.InfoBoxResample.set_text(text)

        def info_UniFilterScoreFunc(self, widget):
            text = ("Information on the score functions for univariate filter can be found on scikit-learn.org")
            self.InfoBoxUnivariateFilterDT.set_text(text)

        def info_UniFilterSearchStratDT(self, widget):
            text = ("Select a search strategy. Percentile selects the best x% of features, k-best the x best features. See scikit-learn.org")
            self.InfoBoxUnivariateFilterDT.set_text(text)

        def info_UniFilterParamDT(self, widget):
            text = ("Enter a value regarding the search strategy: "
                    "If percentile: percent of features to keep, e.g. 50. "
                    "If k-best: amount of features to keep, e.g. 4.")
            self.InfoBoxUnivariateFilterDT.set_text(text)

        def info_FeaturesRFEDT(self, widget):
            text = ("Enter an integer or \"automatic\". Integer selects the entered number of features, \"automatic\" finds optimal number by itself, only automatic supports cross-validation. Information can be found on scikit-learn.org (RFE). ")
            self.InfoBoxMultivariateEmbeddedDT.set_text(text)

        def info_RFE_CV_DT(self, widget):
            text = ("Select the type of cross-validation. Cross-validation information can be found on scikit-learn.org")
            self.InfoBoxMultivariateEmbeddedDT.set_text(text)

        #Template
        """
        def (self, widget):
            text = ("")
            set_text(self, text)
        """

    #executive functions
    def compute_FB(self, widget):
        def saveCallbackFB(OwnlagType):
            'save entries in the "SharedVariables.py" file which will be loaded from the executive scripts "DataTuning" and "ModelTuning"'
            #Values DataTuning
            SV.FixImport = False
            SV.EstimatorWrapper = rf_predictor
            SV.MinIncrease = 0.005
            SV.Resample = False
            SV.InitManFeatureSelect = False
            SV.StandardScaling = False
            SV.RobustScaling = True
            SV.NoScaling = False
            SV.TimeSeriesPlot = False
            SV.ManSelect = False
            SV.Cross_auto_cloud_correlation_plotting = False
            SV.DifferenceCreate = self.checkbox_Difference.get_value()
            SV.FeaturesDifference = True
            SV.ManFeaturelagCreate = False
            SV.AutoFeaturelagCreate = self.checkbox_FeatureLag.get_value()
            if SV.AutoFeaturelagCreate == True:
                SV.MinFeatureLag = int(self.dialog_featurelag.get_field('minFeatureLag').get_value())
                SV.MaxFeatureLag = int(self.dialog_featurelag.get_field('maxFeatureLag').get_value())
            SV.ManFeatureSelect = False
            SV.LowVarianceFilter = False
            SV.ICA = False
            SV.UnivariateFilter = False
            rf = RandomForestRegressor(max_depth=10e17, random_state=0)
            SV.EstimatorEmbedded = rf
            SV.EmbeddedFeatureSelectionThreshold = False
            SV.RecursiveFeatureSelection = False
            SV.WrapperRecursiveFeatureSelection = False

            #Values ModelTuning
            SV.NameOfSubTest = self.txt_NameOfSubTest.get_value()
            SV.StartTraining = self.txt_StartDateTraining.get_value()
            SV.EndTraining = self.txt_EndDateTraining.get_value()
            SV.StartTesting = self.txt_StartDateTesting.get_value()
            SV.EndTesting = self.txt_EndDateTesting.get_value()
            SV.GlobalMaxEval_HyParaTuning = int(self.txt_HyperBayesEval.get_value())
            SV.GlobalCV_MT = 3
            SV.GlobalShuffle = False
            SV.MaxEval_Bayes = int(self.txt_FinalBayesEval.get_value())
            SV.Model_Bayes = "Baye"
            SV.EstimatorEmbedded_FinalBaye = RandomForestRegressor(max_depth=10e17, random_state=0)

            #Overall variables
            SV.NameOfData = self.NameOfDataFB.get_value()
            SV.ColumnOfSignal = int(self.txt_ColumnOfSignal.get_value())

            if OwnlagType == "NoOL":
                SV.NameOfExperiment = "NoOL"
                SV.WrapperParams = [None, None, None, False]
                SV.ManOwnlagCreate = False
                SV.AutomaticTimeSeriesOwnlagConstruct = False
                SV.GlobalRecu = False
            if OwnlagType == "TSOL":
                SV.NameOfExperiment = "TSOL"
                SV.WrapperParams = [None, None, None, True]
                SV.ManOwnlagCreate = False
                SV.AutomaticTimeSeriesOwnlagConstruct = True
                SV.GlobalRecu = True
            if OwnlagType == "InertiaOL":
                SV.NameOfExperiment = "InertiaOL"
                SV.WrapperParams = [None, None, None, True]
                SV.ManOwnlagCreate = True
                SV.OwnLag = [1]
                SV.AutomaticTimeSeriesOwnlagConstruct = False
                SV.GlobalRecu = True
            if OwnlagType == "PeriodOL":
                SV.NameOfExperiment = "PeriodOL"
                SV.WrapperParams = [None, None, None, True]
                SV.ManOwnlagCreate = True
                SV.OwnLag = [int(self.dialog_periodOL.get_field('periodOL').get_value())]
                SV.AutomaticTimeSeriesOwnlagConstruct = False
                SV.GlobalRecu = True

        def exec():
            'executes the two executive scripts'
            DataTuning.main()
            ModelTuning.main_FinalBayes()

        def save_and_execute(OwnlagType):
            saveCallbackFB(OwnlagType=OwnlagType)
            exec()

        self.InfoFB.set_text('Check out the python console!')

        if self.checkbox_NoOL.get_value() == True:  # the no ownlag set shall be considered
            save_and_execute(OwnlagType="NoOL")
            #p1 = Process(target=save_and_execute, args=('NoOL',))
        if self.checkbox_InertiaOL.get_value() == True:
            save_and_execute(OwnlagType="InertiaOL")
            #p2 = Process(target=save_and_execute, args=('InertiaOL',))
        if self.checkbox_TSOL.get_value() == True:
            save_and_execute(OwnlagType="TSOL")
            #p3 = Process(target=save_and_execute, args=('TSOL',))
        if self.checkbox_PeriodOL.get_value() == True:
            save_and_execute(OwnlagType="PeriodOL")
            #p4 = Process(target=save_and_execute, args=('PeriodOL',))
        '''if __name__ == '__main__':
            p1.start()
            p2.start()
            p3.start()
            p4.start()
            p1.join()
            p2.join()
            p3.join()
            p4.join()''' #Todo: Multiprocessing was tried. But it seems that there is a problem with launching multiple processes from a GUI

    def compute_DT(self, widget):
        def saveCallbackDT():
            'save entries in the "SharedVariables.py" file which will be loaded from the executive scripts "DataTuning"'
            #Values DataTuning
            SV.FixImport = False
            SV.NameOfData = self.NameOfDataDT.get_value()
            SV.NameOfExperiment = self.NameOfExperimentDT.get_value()
            SV.ColumnOfSignal = int(self.ColumnOfSignalDT.get_value())

            # Preprocessing
            SV.NaNDealing = self.NaNDealing.get_key()
            SV.Resample = self.ResampleDT.get_value()
            if SV.Resample==True:
                SV.Resolution = self.ResolutionDT.get_value()
                operatordic = {"mean": np.mean, "sum": np.sum, "median": np.median}  # define the operators here
                SV.WayOfResampling = convert_string_to_list(self.WayOfResamplingDT.get_value(), operatordic)
            SV.InitManFeatureSelect = self.InitManFeatureSelectDT.get_value()
            if SV.InitManFeatureSelect==True:
                SV.InitFeatures = convert_string_to_list(self.InitFeatures.get_text(), int)
            def scalerconvert(key):
                SV.StandardScaling = False #set all false
                SV.RobustScaling = False
                SV.NoScaling = False
                if "Standard" == key:
                    SV.StandardScaling = True
                if "Robust" == key:
                    SV.RobustScaling = True
                if "No" == key:
                    SV.NoScaling = True
            scalerconvert(self.Scaler.get_key())
            #Period selection
            SV.TimeSeriesPlot = self.TimeSeriesPlotDT.get_value()
            SV.ManSelect = self.ManPeriodSelectionDT.get_value()
            if SV.ManSelect == True:
                SV.StartDate = self.StartDateDT.get_value()
                SV.EndDate = self.EndDateDT.get_value()
            #Feature construction
            SV.Cross_auto_cloud_correlation_plotting = self.CrossAutoPlotDT.get_value()
            if SV.Cross_auto_cloud_correlation_plotting == True:
                SV.LagsToBePlotted = int(self.CrossAutoPlotLagsDT.get_value())
            SV.DifferenceCreate = self.DifferenceDT.get_value()
            if SV.DifferenceCreate == True:
                SV.FeaturesDifference = self.FeaturesDifferenceAllDT.get_value()
                if SV.FeaturesDifference != True:
                    SV.FeaturesDifference = convert_string_to_list(self.FeaturesDifferenceDT.get_value(), int)
            SV.ManOwnlagCreate = self.ManOwnLagDT.get_value()
            if SV.ManOwnlagCreate == True:
                SV.OwnLag = convert_string_to_list(self.OwnLagsDT.get_value(), int)
            SV.AutomaticTimeSeriesOwnlagConstruct = self.AutoOwnLagDT.get_value()
            SV.ManFeaturelagCreate = self.ManFeatureLagDT.get_value()
            if SV.ManFeaturelagCreate == True:
                ListOfLists = []
                ListOfStrings = convert_string_to_list(self.ManFeatureLagFeaturesDT.get_value(), str, ";")
                for x in ListOfStrings:
                    ListPerFeature = convert_string_to_list(x, int)
                    ListOfLists.append(ListPerFeature)
                SV.FeatureLag = ListOfLists
            SV.AutoFeaturelagCreate = self.AutoFeatureLagDT.get_value()
            if SV.AutoFeaturelagCreate == True:
                SV.MinFeatureLag = int(self.MinFeatureLagDT.get_value())
                SV.MaxFeatureLag = int(self.MaxFeatureLagDT.get_value())

            #Todo: überlegen was ich mit wrapper model und embedded model mache (rf einfach festsetzen?)
            SV.ManFeatureSelect = self.ManFeatureSelectDT.get_value()
            if SV.ManFeatureSelect == True:
                SV.FeatureSelect = convert_string_to_list(self.ManFeatureSelectFeaturesDT.get_value(), int)
            SV.LowVarianceFilter = self.LowVarFilterDT.get_value()
            if SV.LowVarianceFilter ==True:
                SV.Threshold_LowVarianceFilter = float(self.VarianceDT.get_value())
            SV.ICA = self.ICADT.get_value()
            SV.UnivariateFilter = self.UnivariateFilterDT.get_value()
            if SV.UnivariateFilter == True:
                SV.Score_func = self.UniFilterScorFuncDT.get_key()
                SV.SearchMode = self.UniFilterSearchStratDT.get_key()
                SV.Param_univariate_filter = float(self.UniFilterParamDT.get_value()) #Todo: review if float works with number of features which is int
            SV.EmbeddedFeatureSelectionThreshold = self.UnivariateEmbeddedDT.get_value()
            if SV.EmbeddedFeatureSelectionThreshold == True:
                SV.Threshold_embedded = float(self.Threshold_embeddedDT.get_value())
            SV.RecursiveFeatureSelection = self.MultivariateEmbeddedDT.get_value()
            if SV.RecursiveFeatureSelection == True:
                SV.N_feature_to_select_RFE = self.FeaturesRFEDT.get_value()
                if SV.N_feature_to_select_RFE != "automatic":
                    SV.N_feature_to_select_RFE = int(self.FeaturesRFEDT.get_value())
                #convert string type keys into functions to be passed to SharedVariables
                DicCVTypes = {"KFold":KFold, "TimeSeriesSplit":TimeSeriesSplit}
                for key in DicCVTypes:
                    if key == self.RFE_CV_DT.get_key():
                        SV.CV_DT = DicCVTypes[key]((int(self.RFE_CVFolds_DT.get_value())))
            SV.WrapperRecursiveFeatureSelection = self.WrapperRecursiveDT.get_value()

            #Set RF as model for embedded and wrapper methods
            rf = RandomForestRegressor(max_depth=10e10, random_state=0)
            SV.EstimatorEmbedded = rf
            SV.EstimatorWrapper = SV.WrapperModels["RF"]
            SV.WrapperParams = [Hyperparametergrids["RF"], None, None, False]
            SV.MinIncrease = 0

        self.InfoMT.set_text('Check out the python console!')
        saveCallbackDT()
        DataTuning.main()

    def compute_MT(self, widget):
        def saveCallbackMT():
            'save entries in the "SharedVariables.py" file which will be loaded from the executive script "ModelTuning"'
            #Values ModelTuning
            SV.NameOfData = self.NameOfDataMT.get_value()
            SV.NameOfExperiment = self.NameOfExperimentMT.get_value()
            SV.NameOfSubTest = self.NameOfSubTestMT.get_value()
            SV.StartTraining = self.StartTrainingMT.get_value()
            SV.EndTraining = self.EndTrainingMT.get_value()
            SV.StartTesting = self.StartTestingMT.get_value()
            SV.EndTesting = self.EndTestingMT.get_value()
            SV.GlobalMaxEval_HyParaTuning = int(self.HyperBayesEvalMT.get_value())
            SV.GlobalRecu = self.RecursiveMT.get_value()
            SV.GlobalShuffle = self.ShuffleMT.get_value()


            #convert string type keys into functions to be passed to SharedVariables
            DicCVTypes = {"KFold":KFold, "TimeSeriesSplit":TimeSeriesSplit}
            for key in DicCVTypes:
                if key == self.CVTypeMT.get_key():
                    SV.GlobalCV_MT = DicCVTypes[key]((int(self.CVFoldsMT.get_value())))

            #convert all ticked checkboxes into an array with the respective model "keys" to be passed to SharedVariables
            Models = []
            CheckBoxDic = {self.MT_ANN:"ANN", self.MT_ANN_grid:"ANN_grid", self.MT_GB:"GB", self.MT_GB_grid:"GB_grid", self.MT_Lasso:"Lasso", self.MT_Lasso_grid:"Lasso_grid", self.MT_Modelselect:"ModelSelection", self.MT_RF:"RF", self.MT_SVR:"SVR", self.MT_SVR_grid:"SVR_grid"} #additional implemented models need to be entered here as well
            for checkboxes in CheckBoxDic:
                if checkboxes.get_value() == True:
                    Models.append(CheckBoxDic[checkboxes])
            SV.OnlyHyPara_Models = Models
            SV.GlobalIndivModel = self.IndivModelMT.get_key()
            if SV.GlobalIndivModel == "byfeature":
                SV.IndivFeature = self.indivfeature.get_value()
                SV.IndivThreshold = float(self.indivfeaturethreshold.get_value())

        self.InfoMT.set_text('Check out the python console!')
        saveCallbackMT()
        ModelTuning.main_OnlyHyParaOpti()

    def compute_PO(self, widget):
        def saveCallbackPO():
            SV.NameOfData = self.NameOfDataPO.get_value()
            SV.NameOfExperiment = self.NameOfExperimentPO.get_value()
            SV.NameOfSubTest = self.NameOfSubTestPO.get_value()
            SV.NameOfOnlyPredict = self.NameOfOnlyPredictPO.get_value()
            SV.StartTesting = self.StartTestingPO.get_value()
            SV.EndTesting = self.EndTestingPO.get_value()
            SV.OnlyPredictRecursive = self.RecursivePO.get_value()
            SV.ValidationPeriod = self.ValidationPeriodPO.get_value()
            if SV.ValidationPeriod == True:
                SV.StartTest_onlypredict = self.dialog_validationperiod.get_field('StartValidationPeriod').get_value()
                SV.EndTest_onlypredict = self.dialog_validationperiod.get_field('EndValidationPeriod').get_value()

        self.InfoPO.set_text('Check out the python console!')
        saveCallbackPO()
        ModelTuning.main_OnlyPredict()

    #dialog executions
    def exec_dialog_periodOL_FB(self, widget, newValue):
        if newValue == True: #only if the checkbox is ticked
            try:# make sure that the dialog opens with the previously entered values and does not overwrite it with the default ones
                self.dialog_periodOL.show(self) #works if the dialog window already exists(function is not called the first time)
            except: #dialog does not exist, hence it is called the first time. The dialog is created below
                self.dialog_periodOL = gui.GenericDialog(title='Period ownlags',
                                                         message='Enter the respective lags for the period ownlags, e.g. 168 to apply a lag of one week if the resolution is hourly (24*7=168) '
                                                                 '''Period ownlag: One or several ownlags lagged by a typical
period, e.g. 24 hours or a week. It is not physically
valid. Use if unknown influences affect the system and
the system is following any kind of repeating periodic
pattern, e.g. a system affected by unknown occupancies
or control schedules. Expected model behavior: Model
copies pattern from, e.g. yesterday, and adds changes to
that.''',
                                        width='500px')
                self.int_periodOL = gui.SpinBox(168, 0, 100000, width="50%", height=20, margin='10px')
                self.dialog_periodOL.add_field_with_label("periodOL", "Number of lags for the period ownlag", self.int_periodOL)
                self.dialog_periodOL.show(self)
        else:
            pass

    def exec_dialog_featurelag_FB(self, widget, newValue):
        if newValue == True:
            try:
                self.dialog_featurelag.show(self)
            except:
                self.dialog_featurelag = gui.GenericDialog(title='Feature lags',
                                                           message='Enter the respective minimum and maximum lag that shall be considered for finding the optimal feature lag of each feature, e.g. lag "16" would be lagged by 4 hours for a resolution of 15min '
                                                                   '''Featurelag One or several lagged values of a feature. It can be
                                                                    physically valid. Use if a feature has a delayed influence
                                                                    on the signal. Expected model behavior: Model is able to
                                                                    grasp the influence of the lagged feature on the signal,
                                                                    e.g. lagged radiator temperature as feature to the room temperature as signal. ''',
                                            width='500px')
                self.int_minFeatureLag = gui.SpinBox(1, 1, 100000, width="50%", height=20, margin='10px')
                self.int_maxFeatureLag = gui.SpinBox(16, 1, 100000, width="50%", height=20, margin='10px')
                self.dialog_featurelag.add_field_with_label("minFeatureLag", "Minimum feature lag", self.int_minFeatureLag)
                self.dialog_featurelag.add_field_with_label("maxFeatureLag", "Maximum feature lag", self.int_maxFeatureLag)
                self.dialog_featurelag.show(self)
        else:
            pass

    def exec_dialog_indivmodelbyfeature_MT(self, widget, ItemName):
        Value = widget.get_key()
        if Value == "byFeature":
            try:
                self.dialog_byfeature.show(self)
            except:
                self.dialog_byfeature = gui.GenericDialog(title='Individual model by feature threshold',
                                                          message='Press ok to save entries. Copy the name of the feature in question and set a threshold. One model will be trained for samples with the feature´s value above the threshold and one for below. ',
                                            width='500px')
                self.indivfeature = gui.TextInput(width="50%", height=20, margin='10px')
                self.indivfeature.set_text("schedule[]")
                self.indivfeaturethreshold = gui.TextInput(width="50%", height=20, margin='10px')
                self.indivfeaturethreshold.set_text("0.5")
                self.dialog_byfeature.add_field_with_label("IndivFeature", "Name of feature", self.indivfeature)
                self.dialog_byfeature.add_field_with_label("IndivThreshold", "Value of threshold", self.indivfeaturethreshold)
                self.dialog_byfeature.show(self)
        else:
            pass

    def exec_dialog_validationperiod_PO(self, widget, newValue):
        if newValue == True:
            try:
                self.dialog_validationperiod.show(self)
            except:
                self.dialog_validationperiod = gui.GenericDialog(title='Validation period', message='Press ok to confirm. Define the period on which the validation through the mean, standard deviation and max error shall be conducted. '
                    'The defined period will be split into periods with the length of the prediction horizon set by the test period. ',
                                            width='500px')
                self.StartValidationPeriod = gui.TextInput(width="50%", height=20, margin='10px')
                self.StartValidationPeriod.set_text('2016-06-12 00:00')
                self.EndValidationPeriod = gui.TextInput(width="50%", height=20, margin='10px')
                self.EndValidationPeriod.set_text('2016-06-20 23:45')
                self.dialog_validationperiod.add_field_with_label("StartValidationPeriod", "Start validation period", self.StartValidationPeriod)
                self.dialog_validationperiod.add_field_with_label("EndValidationPeriod", "End validation period", self.EndValidationPeriod)
                self.dialog_validationperiod.show(self)
        else:
            pass

    def exec_dialog_Resample(self, widget, Value):
        if Value == True:  # only if the checkbox is ticked
            try:  # make sure that the dialog opens with the previously entered values and does not overwrite it with the default ones
                self.dialog_Resample.show(self)  # works if the dialog window already exists(function is not called the first time)
            except:  # dialog does not exist, hence it is called the first time. The dialog is created below
                self.dialog_Resample = gui.GenericDialog(title='Resample', width='500px')

                Container = gui.Widget(width=500, margin='0px auto',
                                             style={'display': 'block', 'overflow': 'hidden', "text-align": "center"})
                self.InfoBoxResample = gui.Label(
                    'Info will appear here when clicking on the label in question. \n Exemplary entries can be found within the entrywidget',
                    width="100%", height="auto", margin='0px')

                Cont1, self.ResolutionDT = text("Resolution", self.info_Resolution, "15min").do()
                Cont2, self.WayOfResamplingDT = text("Way of resampling", self.info_WayOfResampling, "mean, sum, mean, mean").do()

                Container.append([self.Infolbl, self.InfoBoxResample])
                Container.append([Cont1, Cont2])

                self.dialog_Resample.add_field("Resample", Container)
                self.dialog_Resample.show(self)
        else:
            pass

    def exec_dialog_InitManFeatureSelect(self, widget, Value):
        if Value == True:  # only if the checkbox is ticked
            try:  # make sure that the dialog opens with the previously entered values and does not overwrite it with the default ones
                self.dialog_InitManFeatureSelect.show(
                    self)  # works if the dialog window already exists(function is not called the first time)
            except:  # dialog does not exist, hence it is called the first time. The dialog is created below
                self.dialog_InitManFeatureSelect = gui.GenericDialog(title='Initial manual feature selection', width='500px')

                Container = gui.Widget(width=500, margin='0px auto',
                                       style={'display': 'block', 'overflow': 'hidden', "text-align": "center"})
                self.InfoBoxInitManFeatureSelect = gui.Label(
                    'Info will appear here when clicking on the label in question. \n Exemplary entries can be found within the entrywidget',
                    width="100%", height="auto", margin='0px')

                Cont1, self.InitFeatures = text("Number of features", self.info_InitFeatures, "1, 4, 5, 6").do()

                Container.append([self.Infolbl, self.InfoBoxInitManFeatureSelect])
                Container.append([Cont1])

                self.dialog_InitManFeatureSelect.add_field("InitManFeatureSelect", Container)
                self.dialog_InitManFeatureSelect.show(self)
        else:
            pass

    def exec_dialog_ManPeriodSelectionDT(self, widget, Value):
        if Value == True:  # only if the checkbox is ticked
            try:  # make sure that the dialog opens with the previously entered values and does not overwrite it with the default ones
                self.dialog_ManPeriodSelectionDT.show(
                    self)  # works if the dialog window already exists(function is not called the first time)
            except:  # dialog does not exist, hence it is called the first time. The dialog is created below
                self.dialog_ManPeriodSelectionDT = gui.GenericDialog(title='Manual period selection', width='500px')

                Container = gui.Widget(width=500, margin='0px auto',
                                       style={'display': 'block', 'overflow': 'hidden', "text-align": "center"})
                self.InfoBoxManPeriodSelectionDT = gui.Label(
                    'Info will appear here when clicking on the label in question. \n Exemplary entries can be found within the entrywidget',
                    width="100%", height="auto", margin='0px')

                Cont1, self.StartDateDT = text("Start", self.info_Date, "2016-06-02 00:00").do()
                Cont2, self.EndDateDT = text("End", self.info_Date, "2016-12-18 23:45").do()

                Container.append([self.Infolbl, self.InfoBoxManPeriodSelectionDT])
                Container.append([Cont1, Cont2])

                self.dialog_ManPeriodSelectionDT.add_field("ManPeriodSelectionDT", Container)
                self.dialog_ManPeriodSelectionDT.show(self)
        else:
            pass

    def exec_dialog_CrossAutoPlot(self, widget, Value):
        if Value == True:  # only if the checkbox is ticked
            try:  # make sure that the dialog opens with the previously entered values and does not overwrite it with the default ones
                self.dialog_CrossAutoPlot.show(
                    self)  # works if the dialog window already exists(function is not called the first time)
            except:  # dialog does not exist, hence it is called the first time. The dialog is created below
                self.dialog_CrossAutoPlot = gui.GenericDialog(title='Period ownlags',
                                                              message='Define how many lags shall be plotted. The lags are the x-axis while the cross- or autocorrelation score is the y-axis.',
                                                              width='500px')
                self.CrossAutoPlotLagsDT = gui.SpinBox(100, 0, 100000, width="50%", height=20, margin='10px')
                self.dialog_CrossAutoPlot.add_field_with_label("Crossautoplot", "Number of lags (X-axis width)",
                                                               self.CrossAutoPlotLagsDT)
                self.dialog_CrossAutoPlot.show(self)
        else:
            pass

    def exec_dialog_DifferenceDT(self, widget, Value):
        if Value == True:  # only if the checkbox is ticked
            try:  # make sure that the dialog opens with the previously entered values and does not overwrite it with the default ones
                self.dialog_DifferenceDT.show(self)  # works if the dialog window already exists(function is not called the first time)
            except:  # dialog does not exist, hence it is called the first time. The dialog is created below
                self.dialog_DifferenceDT = gui.GenericDialog(title='Difference', message='List of features of which a differenced time series shall be constructed. Enter the column number of the features you want to difference, separated by a comma (start to count from 0 with first Column after Index, don´t count the column of signal)', width='500px')
                Cont_1, self.FeaturesDifferenceDT = text("Features", self.nothing, "2, 3, 7").do()
                Cont_2, self.FeaturesDifferenceAllDT = checkbox("Select all features", self.nothing, False).do()
                self.dialog_DifferenceDT.add_field("FeaturesDifferenceALLDT", Cont_2)
                self.dialog_DifferenceDT.add_field("FeaturesDifferenceDT", Cont_1)
                self.dialog_DifferenceDT.show(self)
        else:
            pass

    def exec_dialog_ManOwnLagDT(self, widget, Value):
        if Value == True:  # only if the checkbox is ticked
            try:  # make sure that the dialog opens with the previously entered values and does not overwrite it with the default ones
                self.dialog_ManOwnLagDT.show(
                    self)  # works if the dialog window already exists(function is not called the first time)
            except:  # dialog does not exist, hence it is called the first time. The dialog is created below
                self.dialog_ManOwnLagDT = gui.GenericDialog(title='Manual ownlag construction',
                                                            message='Enter the respective ownlags separated by a comma. One lag is as long as the resolution of your preprocessed data. To have your signal lagged by e.g. 3 and also 6 time steps enter \"3, 6\"', width='500px')
                Cont_1, self.OwnLagsDT = text("List of ownlags", self.nothing, "1, 24, 168").do()
                self.dialog_ManOwnLagDT.add_field("ManOwnLagDT", Cont_1)
                self.dialog_ManOwnLagDT.show(self)
        else:
            pass

    def exec_dialog_AutoFeatureLagDT(self, widget, newValue):
        if newValue == True:
            try:
                self.dialog_featurelag_DT.show(self)
            except:
                self.dialog_featurelag_DT = gui.GenericDialog(title='Feature lags',
                                                              message='Set the respective minimum and maximum lag that shall be considered for finding the optimal feature lag of each feature , e.g. lag 16 would be lagged by 4 hours with a resolution 15min',
                                            width='500px')
                self.MinFeatureLagDT = gui.SpinBox(1, 1, 100000, width="50%", height=20, margin='10px')
                self.MaxFeatureLagDT = gui.SpinBox(16, 1, 100000, width="50%", height=20, margin='10px')
                self.dialog_featurelag_DT.add_field_with_label("minFeatureLag", "Minimum feature lag", self.MinFeatureLagDT)
                self.dialog_featurelag_DT.add_field_with_label("maxFeatureLag", "Maximum feature lag", self.MaxFeatureLagDT)
                self.dialog_featurelag_DT.show(self)
        else:
            pass

    def exec_dialog_ManFeatureLagDT(self, widget, Value):
        if Value == True:  # only if the checkbox is ticked
            try:  # make sure that the dialog opens with the previously entered values and does not overwrite it with the default ones
                self.dialog_ManFeatureLagDT.show(self)  # works if the dialog window already exists(function is not called the first time)
            except:  # dialog does not exist, hence it is called the first time. The dialog is created below
                self.dialog_ManFeatureLagDT = gui.GenericDialog(title='Manual feature lag construction', message='Enter as many lags per feature as wished, e.g. for a data with 6 columns 1/1,2//24//1 ; each feature is separated by a "/" within each feature the different lags are separated by a ",". While counting the features the signal column needs to be included, starting with first entry till "/" = first column, the entered number are the respective lags. For the column of signal enter anything, it won´t be used anyways. ', width='500px')
                self.ManFeatureLagFeaturesDT = gui.TextInput(width="50%", height=20, margin='10px')
                self.ManFeatureLagFeaturesDT.set_text("1,2,3,24/5////8/")
                self.dialog_ManFeatureLagDT.add_field_with_label("ManFeatureLagDT", "List of feature lags", self.ManFeatureLagFeaturesDT)
                self.dialog_ManFeatureLagDT.show(self)
        else:
            pass

    def exec_dialog_ManFeatureSelectDT(self, widget, Value):
        if Value == True:  # only if the checkbox is ticked
            try:  # make sure that the dialog opens with the previously entered values and does not overwrite it with the default ones
                self.dialog_ManFeatureSelectDT.show(
                    self)  # works if the dialog window already exists(function is not called the first time)
            except:  # dialog does not exist, hence it is called the first time. The dialog is created below
                self.dialog_ManFeatureSelectDT = gui.GenericDialog(title='Manual feature selection',
                                                            message='Enter the column number of the features you want to keep, separated by a comma. (Start to count from 0 with first Column after Index, Column of signal needs to be counted, but will be kept in any case)',
                                                            width='500px')
                Cont_1, self.ManFeatureSelectFeaturesDT = text("Features", self.nothing, "0, 1, 4, 6").do()
                self.dialog_ManFeatureSelectDT.add_field("ManFeatureSelectDT", Cont_1)
                self.dialog_ManFeatureSelectDT.show(self)
        else:
            pass

    def exec_dialog_LowVarFilterDT(self, widget, Value):
        if Value == True:  # only if the checkbox is ticked
            try:  # make sure that the dialog opens with the previously entered values and does not overwrite it with the default ones
                self.dialog_LowVarFilterDT.show(
                    self)  # works if the dialog window already exists(function is not called the first time)
            except:  # dialog does not exist, hence it is called the first time. The dialog is created below
                self.dialog_LowVarFilterDT = gui.GenericDialog(title='Low variance filter',
                                                                   message='removes all features with a lower variance than the stated threshold, variance is calculated with scaled data (if a scaler is used, regularly only features that are constant have a small variance)',
                                                                   width='500px')
                Cont_1, self.VarianceDT = text("Variance threshold", self.nothing, "0.1").do()
                self.dialog_LowVarFilterDT.add_field("LowVarFilterDT", Cont_1)
                self.dialog_LowVarFilterDT.show(self)
        else:
            pass

    def exec_dialog_UnivariateEmbeddedDT(self, widget, Value):
        if Value == True:  # only if the checkbox is ticked
            try:  # make sure that the dialog opens with the previously entered values and does not overwrite it with the default ones
                self.dialog_UnivariateEmbeddedDT.show(self)  # works if the dialog window already exists(function is not called the first time)
            except:  # dialog does not exist, hence it is called the first time. The dialog is created below
                self.dialog_UnivariateEmbeddedDT = gui.GenericDialog(title='Embedded feature selection by threshold',
                                                               message='Enter the respective feature importance threshold (for random forest between 0 and 1). All features with a feature importance below will be deleted. '
                                                                       'Enter "mean" or "median" for the threshold to be set the mean or median of all feature importances. '
                                                                       'See scikit-learn.org (sklearn.feature_selection.SelectFromModel) for more information.', width='500px')
                Cont_1, self.Threshold_embeddedDT = text("Threshold", self.nothing, "mean").do()
                self.dialog_UnivariateEmbeddedDT.add_field("UnivariateEmbeddedDT", Cont_1)
                self.dialog_UnivariateEmbeddedDT.show(self)
        else:
            pass

    def exec_dialog_UnivariateFilterDT(self, widget, Value):
        if Value == True:  # only if the checkbox is ticked
            try:  # make sure that the dialog opens with the previously entered values and does not overwrite it with the default ones
                self.dialog_UnivariateFilterDT.show(self)  # works if the dialog window already exists(function is not called the first time)
            except:  # dialog does not exist, hence it is called the first time. The dialog is created below
                self.dialog_UnivariateFilterDT = gui.GenericDialog(title='Univariate filter', width='500px')

                Container = gui.Widget(width=500, margin='0px auto',
                                       style={'display': 'block', 'overflow': 'hidden', "text-align": "center"})
                self.InfoBoxUnivariateFilterDT = gui.Label(
                    'Info will appear here when clicking on the label in question. \n Exemplary entries can be found within the entrywidget',
                    width="100%", height="auto", margin='0px')

                Cont1, self.UniFilterScorFuncDT = dropdown("Type of score function", self.info_UniFilterScoreFunc, {mutual_info_regression:"mutual information regression", f_regression:"f regression"}).do()
                Cont2, self.UniFilterSearchStratDT = dropdown("Search strategy", self.info_UniFilterSearchStratDT, {"percentile":"percentile", "k_best":"k-best"}).do()
                Cont3, self.UniFilterParamDT = text("Parameter for search strategy", self.info_UniFilterParamDT, "").do()

                Container.append([self.Infolbl, self.InfoBoxUnivariateFilterDT])
                Container.append([Cont1, Cont2, Cont3])

                self.dialog_UnivariateFilterDT.add_field("UnivariateFilterDT", Container)
                self.dialog_UnivariateFilterDT.show(self)
        else:
            pass

    def exec_dialog_MultivariateEmbeddedDT(self, widget, Value):
        if Value == True:  # only if the checkbox is ticked
            try:  # make sure that the dialog opens with the previously entered values and does not overwrite it with the default ones
                self.dialog_MultivariateEmbeddedDT.show(
                    self)  # works if the dialog window already exists(function is not called the first time)
            except:  # dialog does not exist, hence it is called the first time. The dialog is created below
                self.dialog_MultivariateEmbeddedDT = gui.GenericDialog(title='Embedded recursive feature elemination',
                                                                       width='500px')

                Container = gui.Widget(width=500, margin='0px auto',
                                       style={'display': 'block', 'overflow': 'hidden', "text-align": "center"})
                self.InfoBoxMultivariateEmbeddedDT = gui.Label(
                    'Info will appear here when clicking on the label in question. \n Exemplary entries can be found within the entrywidget',
                    width="100%", height="auto", margin='0px')

                Cont1, self.FeaturesRFEDT = text("Number of features", self.info_FeaturesRFEDT, "automatic").do()

                Cont2 = gui.Widget(width='100%', layout_orientation=gui.Widget.LAYOUT_HORIZONTAL,
                                       margin='0px',
                                       style={'display': 'block', 'overflow': 'auto',
                                              "text-align": "left"})
                lbl_CVDT = gui.Label("Cross-validation", width="40%", height=20, margin="10px")
                lbl_CVDT.set_on_click_listener(self.info_RFE_CV_DT)
                self.RFE_CV_DT = gui.DropDown(width="40%", height=20, margin='10px')
                entries = {"KFold": "KFold", "TimeSeriesSplit": "TimeSeriesSplit"}
                for key in entries:
                    self.RFE_CV_DT.append(entries[key], key)
                self.RFE_CV_DT.select_by_key(next(iter(entries)))  # select the first item of the entries dictionary as initial value
                self.RFE_CVFolds_DT = gui.SpinBox(3, 1, 100, width="6%", height=20, margin="10px")
                Cont2.append([lbl_CVDT, self.RFE_CV_DT, self.RFE_CVFolds_DT])

                Container.append([self.Infolbl, self.InfoBoxMultivariateEmbeddedDT])
                Container.append([Cont1, Cont2])

                self.dialog_MultivariateEmbeddedDT.add_field("MultivariateEmbeddedDT", Container)
                self.dialog_MultivariateEmbeddedDT.show(self)
        else:
            pass

    #template
    '''
    def exec_dialog_$(self, widget, Value):
        if Value == True: #only if the checkbox is ticked
            try:# make sure that the dialog opens with the previously entered values and does not overwrite it with the default ones
                self.dialog_$.show(self) #works if the dialog window already exists(function is not called the first time)
            except: #dialog does not exist, hence it is called the first time. The dialog is created below
                self.dialog_$ = gui.GenericDialog(title='$', width='500px')

                Container = gui.Widget(width=500, margin='0px auto',
                                             style={'display': 'block', 'overflow': 'hidden', "text-align": "center"})
                self.InfoBox$ = gui.Label(
                    'Info will appear here when clicking on the label in question. \n Exemplary entries can be found within the entrywidget',
                    width="100%", height="auto", margin='0px')

                Cont1, self.Trial = text("Name of data", self.info_NameOfData, "AHU1").do()


                Container.append([self.Infolbl, self.InfoBox$])
                Container.append([])

                self.dialog_$.add_field("$", Container)
                self.dialog_$.show(self)
        else:
            pass
    '''


start(AutomatedTraining, address='0.0.0.0', port=8081, multiple_instance=True, enable_file_cache=True, update_interval=0.9, start_browser=True)
#Todo: if docker address = 0.0.0.1 if normal = 127.0.0.1

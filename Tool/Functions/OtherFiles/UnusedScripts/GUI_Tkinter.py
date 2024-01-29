from tkinter import *

import DataTuning
import SharedVariables as SV
import FinalBayesOpt

print("Import in GUI_Tkinter.py done")
master = Tk()

#05.06.2018 progress stopped, continued with the Remi GUI


def saveCallback2(OwnlagType): #save entries in an txt file which will be loaded from the executive scripts globalvariables and FinalBayesOpt
    #Overall Variables
    SV.name_of_raw_data = NameOfData.get()


    #Variables for Data Tuning
    #SVF.ColumnOfSignal = ColumnSignal.get()
    #SVF.AutoFeatureLagLags = AutoFeatureLagLags.get()

    #Variables for Model Tuning
    SV.name_of_model_tuning_experiment = NameOfSubTest.get()
    SV.start_train_val = StartTraining.get()
    SV.stop_train_val = EndTraining.get()
    SV.start_test = StartTesting.get()
    SV.end_test = EndTesting.get()
    SV.start_train_val = StartTraining.get()

    if OwnlagType == "NoOL":
        SV.name_of_data_tuning_experiment = "NoOL"
        SV.wrapper_params = [None, None, None, False]
        SV.create_manual_target_lag = False
        SV.create_automatic_timeseries_target_lag = False
        SV.GlobalRecu = False
    if OwnlagType == "TSOL":
        SV.name_of_data_tuning_experiment = "TSOL"
        SV.wrapper_params = [None, None, None, True]
        SV.create_manual_target_lag = False
        SV.create_automatic_timeseries_target_lag = True
        SV.GlobalRecu = True
    if OwnlagType == "InertiaOL":
        SV.name_of_data_tuning_experiment = "InertiaOL"
        SV.wrapper_params = [None, None, None, True]
        SV.create_manual_target_lag = True
        SV.target_lag = [1]
        SV.create_automatic_timeseries_target_lag = False
        SV.GlobalRecu = True
    if OwnlagType == "PeriodOL":
        SV.name_of_data_tuning_experiment = "PeriodOL"
        SV.wrapper_params = [None, None, None, True]
        SV.create_manual_target_lag = True
        SV.target_lag = [672]
        SV.create_automatic_timeseries_target_lag = False
        SV.GlobalRecu = True

def exec():
    DataTuning.main()
    FinalBayesOpt.main()

def run_computation():
    if NoOL.get() == 1: #the no ownlag set shall be considered
        saveCallback2("NoOL")
        exec()
    if InertiaOL.get() == 1:
        saveCallback2("InertiaOL")
        exec()
    if TSOL.get() == 1:
        saveCallback2("TSOL")
        exec()
    if PeriodOL.get() == 1:
        saveCallback2("PeriodOL")
        exec()



#Defining the folder where results shall be safed
row = 0 #updating rowcounter
Label(master, text="Defining the folder for results:").grid(row=row, columnspan=2, sticky=W)

row += 1 #updating rowcounter
Label(master, text="Name of Data").grid(row=row, sticky=W)
NameOfData = Entry(master)
NameOfData.insert(1, "CodeTestingGUI")
NameOfData.grid(row=row, column=1)

row += 1 #updating rowcounter
Label(master, text="Name of Subtest").grid(row=row, sticky=W)
NameOfSubTest = Entry(master)
NameOfSubTest.insert(1, "Test1")
NameOfSubTest.grid(row=row, column=1)



#Defining the periods
row += 1 #updating rowcounter
Label(master, text="Date entries:").grid(row=row, column=0, sticky=W)
Label(master, text="year-month-day hour:minutes").grid(row=row, column=1, sticky=W)

row += 1 #updating rowcounter
Label(master, text="Start training:").grid(row=row, sticky=W)
StartTraining = Entry(master)
StartTraining.insert(1, "2016-06-02 00:00")
StartTraining.grid(row=row, column=1)

row += 1 #updating rowcounter
Label(master, text="End training:").grid(row=row, sticky=W)
EndTraining = Entry(master)
EndTraining.insert(1, "2016-06-09 00:00")
EndTraining.grid(row=row, column=1)

row += 1 #updating rowcounter
Label(master, text="Start testing:").grid(row=row, sticky=W)
StartTesting = Entry(master)
StartTesting.insert(1, "2016-06-09 00:00")
StartTesting.grid(row=row, column=1)

row += 1 #updating rowcounter
Label(master, text="End testing:").grid(row=row, sticky=W)
EndTesting = Entry(master)
EndTesting.insert(1, "2016-06-16 00:00")
EndTesting.grid(row=row, column=1)



#Define ownlags that shall be considered
row += 1#initiating rowcounter
Label(master, text="Ownlags to be considered: ").grid(row=row, sticky=W)

row += 1 #updating rowcounter
NoOL = IntVar()
Checkbutton(master, text="No ownlag", variable=NoOL).grid(row=row, sticky=W)

PeriodOL = IntVar()
Checkbutton(master, text="Period ownlag", variable=PeriodOL).grid(row=row, column=1, sticky=W)

row += 1 #updating rowcounter
TSOL = IntVar()
Checkbutton(master, text="Time series ownlag", variable=TSOL).grid(row=row, sticky=W)

InertiaOL = IntVar()
Checkbutton(master, text="Inertia ownlag", variable=InertiaOL).grid(row=row, column=1, sticky=W)

#Buttons
row += 1 #updating rowcounter
Button(master, text='Quit', command=master.quit).grid(row=row, sticky=W, pady=4)

#row += 1 #updating rowcounter
Button(master, text='Run computation', command=run_computation).grid(row=row, column=1, sticky=W, pady=4)

if __name__ == "__main__":
    mainloop()

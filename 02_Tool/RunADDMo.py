"""
Created on Tue Feb 24 2022

@author: lma-afu
Change options in SharedVariables and run both DataTuning and ModelTuning
"""

import SharedVariables as SV
import DataTuning
import ModelTuning


def change_shared_variables(NameOfData, NameOfExperiment, NameOfSubTest, GlobalMaxEval_HyParaTuning, MaxEval_Bayes,
                            Model_Bayes):
    """Change options in SharedVariables.py

    Parameters
    ----------
    NameOfData : str
        Set name of the folder where the experiments shall be saved, e.g. the name of the observed data
    NameOfExperiment : str
        Set name of the experiments series
    NameOfSubTest : str
        Results will be stored in this folder
    GlobalMaxEval_HyParaTuning  : int
        sets the number of evaluations done by the bayesian optimization for each "tuned training" to find the best
        Hyperparameter, each evaluation is training and testing with cross-validation for one hyperparameter setting
    MaxEval_Bayes : int
        Number of iterations the bayesian optimization should do for selecting NumberofFeatures,
        IndivModel, BestModel , the less the less quality but faster
    Model_Bayes : str
        "SVR","ANN","GB","RF","Lasso" - choose a model for bayesian optimization (RF is by far the fastest)
        "ModelSelection" - bayesian optimization is done with the score of the best model
        (hence in each iteration all models are calculated)
        "Baye" - models are chosen through bayesian optimization as well (consider higher amount of Max_eval_bayes
    Returns
    -------
    """

    SV.NameOfData = NameOfData
    SV.NameOfExperiment = NameOfExperiment
    SV.NameOfSubTest = NameOfSubTest
    SV.GlobalMaxEval_HyParaTuning = GlobalMaxEval_HyParaTuning
    SV.MaxEval_Bayes = MaxEval_Bayes
    SV.Model_Bayes = Model_Bayes


if __name__ == '__main__':

    # Ganze Jahre für thermische Bedarfe
    StartTraining = '2020-01-01 00:00'
    EndTraining = "2020-06-30 23:45"
    StartTesting = "2020-07-01 00:00"
    EndTesting = "2020-12-31 23:45:00"

    SV.StandardScaling = False  # Für Reihen mit vielen Nullen
    SV.RobustScaling = True  # für alle sonstigen daten

    NameOfData = "ECOS2023"
    NameOfExperiment = "AdaptedData_all_4preds_500estimators"

    NameOfSubTest = "RFbay_120_ACC"
    GlobalMaxEval_HyParaTuning = 120
    MaxEval_Bayes = 3
    Model_Bayes = "RF"

    num_of_experiments = 3

    # Set True if DataTuning shall be run before ModelTuning
    run_datatuning = True

    if run_datatuning:
        first_num_of_experiment = 0
    else:
        first_num_of_experiment = 1
        num_of_experiments += 1

    for experiment in range(first_num_of_experiment, num_of_experiments):
        change_shared_variables(NameOfData, NameOfExperiment, NameOfSubTest + str(experiment),
                                GlobalMaxEval_HyParaTuning, MaxEval_Bayes, Model_Bayes)
        if experiment == 0:
            DataTuning.main()
            # choices: ModelTuning.main_FinalBayes(), ModelTuning.main_OnlyHyParaOpti(), main_OnlyPredict()
            ModelTuning.main_OnlyHyParaOpti()
        else:
            ModelTuning.main_OnlyHyParaOpti()

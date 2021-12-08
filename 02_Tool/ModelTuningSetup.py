import os

from sklearn.externals import joblib

import SharedVariablesFunctions as SVF

class ModelTuningSetup:
    """
    Object that stores all the setup and user input information of Model Tuning

    # -----------------------Global Variables-------------------------------

    RootDir = "Empty"
    PathToData = "Empty"
    ResultsFolder = "Empty"
    PathToPickles = "Empty"
    NameOfData = "AHU Data1"
    NameOfExperiment = "NoOL"

    # -----------------------Model Tuning Variables-------------------------------

    NameOfSubTest = "Empty"
    StartTraining = '2016-08-01 00:00'
    EndTraining = '2016-08-14 23:45'
    StartTesting = '2016-08-15 00:00'
    EndTesting = '2016-08-16 23:45'

    # -- Set global variables, those variables are for the BlackBox models themselves not for the final bayesian optimization --

    GlobalMaxEval_HyParaTuning = 2  # Sets the number of evaluations done by the bayesian optimization for each "tuned training" to find the best Hyperparameter, each evaluation is training and testing with cross-validation for one hyperparameter setting
    GlobalCV_MT = 3  # Enter any cross-validation method from scikit-learn or any self defined or from elsewhere.
    GlobalRecu = True  # (Boolean) this sets whether the it shall be forecasted recursive or not
    GlobalShuffle = True

    OnlyHyPara_Models = ["ModelSelection"]  # array of the blackboxes you want to use
    # Possible entries: ["SVR", "RF", "ANN", "GB", "Lasso", "SVR_grid", "ANN_grid", "RF_grid", "GB_grid", "Lasso_grid"]
    # ["ModelSelection"] uses all bayesian models (those without _grid) and returns the best

    # -- Settings for regular training without final bayesian optimization (without "automation) -----------------------

    GlobalIndivModel = "hourly"  # "week_weekend"; "hourly"; "No"; "byFeature"
    if GlobalIndivModel == "byFeature":
        F = "schedule[]"  # copy the name of feature here
        IndivThreshold = 0.5  # state the threshold at which value of that feature the data frame shall be split

    # -----------------------Only Predict Variables-------------------------------

        # This is to use after training the models, hence it won´t produce results if there is no trained model saved already(which is done automatically if training one)
        # You define the trained model you want to load through nameofexperiment and NameOfSubTest and the time you want to predict through __StartDateTest and __EndDateTest
        # of course it is necessary that the models have been trained before in the respective nameofdata and nameofexperiment and NameOfSubTest combination

        NameOfOnlyPredict = "TestNew5"  # use different names if you want to use several only_predicts on the same trained models
        OnlyPredictRecursive = True

        ValidationPeriod = True
        #Set False to have the prediction error on the whole data period (train and test),
            set True to define a test period by yourself(example the whole outhold data).
            With the difference between StartTesting and EndTesting the required prediction horizon is set
            (this period (StartTesting till EndTesting) is also the one being plotted and analysed with the regular measures)
            The defined test period is then split into periods with the length of "horizon", for each horizon the prediction
            error is computed. Of those errors the "mean", "standard deviation" and the "max error" are computed
            (see "Automated Data Driven Modeling of Building Energy Systems via Machine Learning Algorithms" by Martin Rätz for more details)

        if ValidationPeriod == True:
            StartTest_onlypredict = '2016-06-09 00:00'
            EndTest_onlypredict = '2016-06-15 00:00'

    # -----------------------Auto Final Bayes Variables-------------------------------

    #Final bayesian optimization finds optimal combination of "Individual Model"&"Features"&"Model"
    #Final bayesian optimization parameter
    MaxEval_Bayes = 5
    Model_Bayes = "Baye"
    # possible entries
    # Max_eval_Bayes = int - Number of iterations the bayesian optimization should do for selecting NumberofFeatures, IndivModel, BestModel , the less the less quality but faster
    # Model= "SVR","ANN","GB","RF","Lasso" - choose a model for bayesian optimization (RF is by far the fastest)
    #        "ModelSelection" - bayesian optimization is done with the score of the best model (hence in each iteration all models are calculated)
    #        "Baye" - models are chosen through bayesian optimization as well (consider higher amount of Max_eval_bayes

    # define and set Estimator which shall be used in for the embedded feature selection
    rf = RandomForestRegressor(max_depth=10e17, random_state=0)  #have to be defined so that they return "feature_importance", more implementation have to be developed
    EstimatorEmbedded_FinalBaye = rf  # e.g. <rf>; set one of the above models to be used

    """

    def __init__(self, **kwargs):

        # -----------------------Global Variables-------------------------------

        self.RootDir = "Empty"
        self.PathToData = "Empty"
        self.ResultsFolder = "Empty"
        self.PathToPickles = "Empty"
        self.NameOfData = "AHU Data1"
        self.NameOfExperiment = "NoOL"
        self.NameOfSignal = "Empty"
        self.ColumnOfSignal = 1

        # -----------------------Model Tuning Variables-------------------------------

        self.NameOfSubTest = "Empty"
        self.StartTraining = '2016-08-01 00:00'
        self.EndTraining = '2016-08-14 23:45'
        self.StartTesting = '2016-08-15 00:00'
        self.EndTesting = '2016-08-16 23:45'

        # Set global variables, those variables are for the BlackBox models themselves not for the final bayesian optimization

        self.GlobalMaxEval_HyParaTuning = 2
        self.GlobalCV_MT = 3
        self.GlobalRecu = True
        self.GlobalShuffle = True

        # Settings for regular training without final bayesian optimization (without "automation)

        self.GlobalIndivModel = "hourly"
        if self.GlobalIndivModel == "byFeature":
            self.IndivFeature = "schedule[]"
            self.IndivThreshold = 0.5
        self.OnlyHyPara_Models = ["ModelSelection"]

        # -----------------------Only Predict Variables-------------------------------

        self.NameOfOnlyPredict = "TestNew5"
        self.OnlyPredictRecursive = True

        self.ValidationPeriod = True
        if self.ValidationPeriod == True:
            self.StartTest_onlypredict = '2016-06-09 00:00'
            self.EndTest_onlypredict = '2016-06-15 00:00'

        # -----------------------Auto Final Bayes Variables-------------------------------

        self.MaxEval_Bayes = 5
        self.Model_Bayes = "Baye"
        self.EstimatorEmbedded_FinalBaye = SVF.rf

    def dump_object(self):
        print("Saving Model Tuning Setup class Object as a pickle in path: '%s'" % os.path.join(self.ResultsFolder,
                                                                                                "ModelTuningSetup.save"))
        # Save the object as a pickle for reuse
        joblib.dump(self, os.path.join(self.ResultsFolder, "ModelTuningSetup.save"))

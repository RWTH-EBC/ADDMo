from core.util.abstract_config import BaseConfig

class ModelTuningSetup(BaseConfig):

    def __init__(self, **kwargs):
        # -----------------------Global Variables-------------------------------

        self.name_of_raw_data: str = "AHU Data1"  # Refer to the raw data connected to this
        self.name_of_data_tuning_experiment: str = "NoOL"  # Refer to the data tuning experiment
        # aka the input data for this model tuning experiment

        self.name_of_model_tuning_experiment: str = "TrialTunedModel"  # Set name of the
        # current experiment

        self.abs_path_to_data: str = r"D:\\04_GitRepos\\addmo-extra\\raw_input_data\\InputData.xlsx"  # Path to the file that has
        # the data

        self.name_of_target:str = "FreshAir Temperature"
        # Name of the target variable

        # -----------------------Model Tuning Variables-------------------------------

        self.start_train_val: str  = "2016-08-01 00:00"
        self.stop_train_val: str  = "2016-08-14 23:45"
        self.start_test: str  = "2016-08-15 00:00"
        self.end_test: str  = "2016-08-16 23:45"

        self.hyperparameter_tuning_type: str  = "optuna"  # grid, optuna, none

        self.iterations_hyperparameter_tuning:int = 2
        self.validation_score_mechanism:str = "cv"
        self.validation_score_splitting: str  = "kfold" # all custom splitters or scikit-learn splitters
        self.validation_score_splitting_kwargs: dict = None
        self.validation_score_metric: str  = "r2"
        self.validation_score_metric_kwargs: dict = None

        self.models: list[str] = ["mlp"]  # "all" or array of the models you want to use

        # # -- Settings for regular training without final bayesian optimization (without "automation) -----------------------
        #
        # self.GlobalIndivModel = "hourly"  # "week_weekend"; "hourly"; "No"; "byFeature"
        # if self.GlobalIndivModel == "byFeature":
        #     self.IndivFeature = "schedule[]"  # copy the name of feature here
        #     self.IndivThreshold = 0.5  # state the threshold at which value of that feature the data frame shall be split

        # # -----------------------Only Predict Variables-------------------------------
        #
        # self.NameOfOnlyPredict = "TestNew5"
        # self.OnlyPredictRecursive = True
        #
        # self.ValidationPeriod = True
        # # Set False to have the prediction error on the whole data period (fit and test),
        # # set True to define a test period by yourself(example the whole outhold data).
        # # With the difference between StartTesting and EndTesting the required prediction horizon is set
        # # (this period (StartTesting till EndTesting) is also the one being plotted and analysed with the regular measures)
        # # The defined test period is then split into periods with the length of "horizon", for each horizon the prediction
        # # error is computed. Of those errors the "mean", "standard deviation" and the "max error" are computed
        # # (see "Automated Data Driven Modeling of Building Energy Systems via Machine Learning Algorithms" by Martin Rätz for more details)
        #
        # if self.ValidationPeriod == True:
        #     self.StartTest_onlypredict = '2016-06-09 00:00'
        #     self.EndTest_onlypredict = '2016-06-15 00:00'
        #
        # # -----------------------Auto Final Bayes Variables-------------------------------
        #
        # # Final bayesian optimization finds optimal combination of "Individual Model"&"Features"&"Model"
        # # Final bayesian optimization parameter
        # self.MaxEval_Bayes = 5
        # self.Model_Bayes = "Baye"
        # # possible entries
        # # Max_eval_Bayes = int - Number of iterations the bayesian optimization should do for selecting NumberofFeatures, IndivModel, BestModel , the less the less quality but faster
        # # Model= "SVR","ANN","GB","RF","Lasso" - choose a model for bayesian optimization (RF is by far the fastest)
        # # "ModelSelection" - bayesian optimization is done with the score_test of the best model (hence in each iteration all models are calculated)
        # # "Baye" - models are chosen through bayesian optimization as well (consider higher amount of Max_eval_bayes
        #
        # # define and set Estimator which shall be used in for the embedded feature selection
        # rf = RandomForestRegressor(max_depth=10e17,
        #                            random_state=0)  # have to be defined so that they return "feature_importance", more implementation have to be developed
        # self.EstimatorEmbedded_FinalBaye = rf  # e.g. <rf>; set one of the above models to be used

    # def dump_object(self):
    #     print(
    #         "Saving Model Tuning Setup class Object as a pickle in path: '%s'"
    #         % os.path.join(self.ResultsFolder, "ModelTuningSetup.save")
    #     )
    #     # Save the object as a pickle for reuse
    #     joblib.dump(self, os.path.join(self.ResultsFolder, "ModelTuningSetup.save"))

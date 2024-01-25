from __future__ import print_function
from math import log
from sklearn.svm import SVR
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import metrics
from Functions.ErrorMetrics import *

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
from hyperopt.pyll import scope
import hyperopt.pyll.stochastic
from sklearn.externals import joblib
import os
import warnings

import SharedVariablesFunctions as SVF
import PredictorDefinitions as PD
import Documentation as Document
import ModelTuningRuntimeResults as MTRR

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def train_predict_selected_models(
    MT_Setup_Object,
    MT_RR_object,
    Models,
    _X_train,
    _Y_train,
    _X_test,
    _Y_test,
    Indexer="IndexerError",
    IndividualModel="Error",
    Documentation=False,
):
    def evaluate(Model, RR_Model_Summary):

        NameOfPredictor = Model.Estimator.__name__
        if IndividualModel == "week_weekend":
            indivweekweekend = indiv_model(
                indiv_splitter_instance=indiv_splitter(week_weekend_splitter),
                Estimator=Model.Estimator,
                Features_train=_X_train,
                Signal_train=_Y_train,
                Features_test=_X_test,
                Signal_test=_Y_test,
                HyperparameterGrid=Model.HyperparameterGrid,
                CV=MT_Setup_Object.GlobalCV_MT,
                Max_evals=MT_Setup_Object.GlobalMaxEval_HyParaTuning,
                Recursive=MT_Setup_Object.GlobalRecu,
            )
            Result_dic = indivweekweekend.main()
        elif IndividualModel == "hourly":
            indivhourly = indiv_model(
                indiv_splitter_instance=indiv_splitter(hourly_splitter),
                Estimator=Model.Estimator,
                Features_train=_X_train,
                Signal_train=_Y_train,
                Features_test=_X_test,
                Signal_test=_Y_test,
                HyperparameterGrid=Model.HyperparameterGrid,
                CV=MT_Setup_Object.GlobalCV_MT,
                Max_evals=MT_Setup_Object.GlobalMaxEval_HyParaTuning,
                Recursive=MT_Setup_Object.GlobalRecu,
            )
            Result_dic = indivhourly.main()
        elif IndividualModel == "byFeature":
            byFeaturesplitter = byfeature_splitter(
                MT_Setup_Object.IndivThreshold,
                MT_Setup_Object.IndivFeature,
                _X_test,
                _X_train,
            )
            indivbyfeature = indiv_model(
                indiv_splitter_instance=indiv_splitter(byFeaturesplitter.splitter),
                Estimator=Model.Estimator,
                Features_train=_X_train,
                Signal_train=_Y_train,
                Features_test=_X_test,
                Signal_test=_Y_test,
                HyperparameterGrid=Model.HyperparameterGrid,
                CV=MT_Setup_Object.GlobalCV_MT,
                Max_evals=MT_Setup_Object.GlobalMaxEval_HyParaTuning,
                Recursive=MT_Setup_Object.GlobalRecu,
            )
            Result_dic = indivbyfeature.main()
        else:
            Result_dic = Model.Estimator(
                Features_train=_X_train,
                Signal_train=_Y_train,
                Features_test=_X_test,
                Signal_test=_Y_test,
                HyperparameterGrid=Model.HyperparameterGrid,
                CV=MT_Setup_Object.GlobalCV_MT,
                Max_evals=MT_Setup_Object.GlobalMaxEval_HyParaTuning,
                Recursive=MT_Setup_Object.GlobalRecu,
            )

        Predicted = Result_dic["prediction"]
        Bestparams = Result_dic["best_params"]
        ComputationTime = Result_dic["ComputationTime"]
        FeatureImportance = Result_dic["feature_importance"]

        Y_test, Y_Predicted = SVF.apply_scaler(
            MT_Setup_Object, Predicted, _Y_test, Indexer
        )
        Scores = SVF.getscores(
            Y_test, Y_Predicted
        )  # Todo: Make possible to set scoring function by yourself

        if (
            Documentation == True
        ):  # only do documentation if Documentation is wished(Documentation is False from beginning, and only in the end set True)

            Document.documentation_model_tuning(
                MT_Setup_Object,
                RR_Model_Summary,
                NameOfPredictor,
                Y_Predicted,
                Y_test,
                _Y_train,
                ComputationTime,
                Scores,
                Model.HyperparameterGridString,
                Bestparams,
                IndividualModel,
                FeatureImportance,
            )

            # Plot Results
            Document.visualization(
                MT_Setup_Object, NameOfPredictor, Y_Predicted, Y_test, Scores
            )

            # only dump if it´s the last best one(marked by Documentation=True)
            MTRR.model_saver(
                Result_dic,
                MT_Setup_Object.ResultsFolderSubTest,
                NameOfPredictor,
                IndividualModel,
            )

        # Return R2 value
        return Scores[0]

    def modelselection():
        # Trains and tests all (bayesian) models and returns the best of them, also saves it in an txtfile.

        Score_SVR = evaluate(BlackBox1, MT_RR_object.SVR_Summary)
        Score_RF = evaluate(BlackBox2, MT_RR_object.RF_Summary)
        Score_ANN = evaluate(BlackBox3, MT_RR_object.ANN_Summary)
        Score_GB = evaluate(BlackBox4, MT_RR_object.GB_Summary)
        Score_Lasso = evaluate(BlackBox5, MT_RR_object.Lasso_Summary)

        Score_list = [0, 1, 2, 3, 4]
        Score_list[0] = Score_SVR
        Score_list[1] = Score_RF
        Score_list[2] = Score_ANN
        Score_list[3] = Score_GB
        Score_list[4] = Score_Lasso

        print(Score_list)
        # Todo: if Scoring function Score max; if Scoring function some error: min
        BestScore = max(Score_list)

        if Score_list[0] == BestScore:
            __BestModel = "SVR"
        if Score_list[1] == BestScore:
            __BestModel = "RF"
        if Score_list[2] == BestScore:
            __BestModel = "ANN"
        if Score_list[3] == BestScore:
            __BestModel = "GB"
        if Score_list[4] == BestScore:
            __BestModel = "Lasso"

        # state best model in txt file
        f = open(
            os.path.join(MT_Setup_Object.ResultsFolderSubTest, "BestModel.txt"), "w+"
        )
        f.write(
            "The best model is %s with an accuracy of %s" % (__BestModel, BestScore)
        )
        f.close()
        return BestScore

    def train_predict(Model):

        # This function is just to "centralize" the fit and infer operations so that additional options can be added easier
        if Model == "SVR":
            Score = evaluate(BlackBox1, MT_RR_object.SVR_Summary)
        if Model == "RF":
            Score = evaluate(BlackBox2, MT_RR_object.RF_Summary)
        if Model == "ANN":
            Score = evaluate(BlackBox3, MT_RR_object.ANN_Summary)
        if Model == "GB":
            Score = evaluate(BlackBox4, MT_RR_object.GB_Summary)
        if Model == "Lasso":
            Score = evaluate(BlackBox5, MT_RR_object.Lasso_Summary)
        if Model == "SVR_grid":
            Score = evaluate(BlackBox6, MT_RR_object.SVR_grid_Summary)
        if Model == "ANN_grid":
            Score = evaluate(BlackBox7, MT_RR_object.ANN_grid_Summary)
        if Model == "GB_grid":
            Score = evaluate(BlackBox8, MT_RR_object.GB_grid_Summary)
        if Model == "Lasso_grid":
            Score = evaluate(BlackBox9, MT_RR_object.Lasso_grid_Summary)
        if Model == "ModelSelection":
            Score = modelselection()

        return Score

    for Model in Models:
        Score = train_predict(Model)

    return Score


class BlackBox:
    'This Class uses the machine learning "predictors" for training, predicting and documentation defined in BlackBoxes.py'

    def __init__(
        self, Estimator, HyperparameterGrid="None", HyperparameterGridString="None"
    ):
        self.Estimator = Estimator
        self.HyperparameterGrid = HyperparameterGrid
        self.HyperparameterGridString = HyperparameterGridString


# ---------------------------------------- Initiate the blackboxes ----------------------------------------------------

# Info: Make sure the HyperparameterGrid is always equal to the HyperparameterGridString for correct documentation

# ------- SVR (BlackBox1)
HyperparameterGrid1 = {
    "C": hp.loguniform("C", log(1e-4), log(1e4)),
    "gamma": hp.loguniform("gamma", log(1e-3), log(1e4)),
    "epsilon": hp.loguniform("epsilon", log(1e-4), log(1)),
}  # with loguniform(-6, 23.025) spans a range from 1e-3 to 1e10
HyperparameterGridString1 = """{"C": hp.loguniform("C", log(1e-4), log(1e4)), "gamma":hp.loguniform("gamma", log(1e-3),log(1e4)), "epsilon":hp.loguniform("epsilon", log(1e-4), log(1))}"""  # set this as a string in order to have a exact"screenshot" of the hyperparametergrid to save it in the summary
BlackBox1 = BlackBox(
    PD.svr_bayesian_predictor, HyperparameterGrid1, HyperparameterGridString1
)

# ------- RF (BlackBox2)
BlackBox2 = BlackBox(PD.rf_predictor, None, None)

# ------- ANN (BlackBox3)
HyperparameterGrid3 = hp.choice(
    "number_of_layers",
    [
        {"1layer": scope.int(hp.qloguniform("1.1", log(1), log(1000), 1))},
        {
            "2layer": [
                scope.int(hp.qloguniform("1.2", log(1), log(1000), 1)),
                scope.int(hp.qloguniform("2.2", log(1), log(1000), 1)),
            ]
        },
        {
            "3layer": [
                scope.int(hp.qloguniform("1.3", log(1), log(1000), 1)),
                scope.int(hp.qloguniform("2.3", log(1), log(1000), 1)),
                scope.int(hp.qloguniform("3.3", log(1), log(1000), 1)),
            ]
        },
    ],
)
HyperparameterGridString3 = """hp.choice("number_of_layers",
                        [
                        {"1layer": scope.int(hp.qloguniform("1.1", log(1), log(1000), 1))},
                        {"2layer": [scope.int(hp.qloguniform("1.2", log(1), log(1000), 1)), scope.int(hp.qloguniform("2.2", log(1), log(1000), 1))]},
                        {"3layer": [scope.int(hp.qloguniform("1.3", log(1), log(1000), 1)), scope.int(hp.qloguniform("2.3", log(1), log(1000), 1)), scope.int(hp.qloguniform("3.3", log(1), log(1000), 1))]}
                        ])"""  # set this as a string in order to have a exact"screenshot" of the hyperparametergrid to save it in the summary
BlackBox3 = BlackBox(
    PD.ann_bayesian_predictor, HyperparameterGrid3, HyperparameterGridString3
)

# ------- GB (BlackBox4)
HyperparameterGrid4 = {
    "n_estimators": scope.int(hp.qloguniform("n_estimators", log(1), log(1e3), 1)),
    "max_depth": scope.int(hp.qloguniform("max_depth", log(1), log(100), 1)),
    "learning_rate": hp.loguniform("learning_rate", log(1e-2), log(1)),
    "loss": hp.choice("loss", ["ls", "lad", "huber", "quantile"]),
}  # if anything except numbers is changed, please change the respective code lines for converting notation style in the gradienboost_bayesian function
HyperparameterGridString4 = """{"n_estimators": scope.int(hp.qloguniform("n_estimators", log(1), log(1e3), 1)), "max_depth": scope.int(hp.qloguniform("max_depth", log(1),log(100), 1)), "learning_rate":hp.loguniform("learning_rate", log(1e-2), log(1)), "loss":hp.choice("loss",["ls", "lad", "huber", "quantile"])}"""  # set this as a string in order to have a exact"screenshot" of the hyperparametergrid to save it in the summary
BlackBox4 = BlackBox(
    PD.gradientboost_bayesian, HyperparameterGrid4, HyperparameterGridString4
)

# ------- Lasso (BlackBox5)
HyperparameterGrid5 = {"alpha": hp.loguniform("alpha", log(1e-10), log(1e6))}
HyperparameterGridString5 = """{"alpha": hp.loguniform("alpha", log(1e-10), log(1e6))}"""  # set this as a string in order to have a exact"screenshot" of the hyperparametergrid to save it in the summary
BlackBox5 = BlackBox(PD.lasso_bayesian, HyperparameterGrid5, HyperparameterGridString5)

# ------- SVR_grid (BlackBox6)
HyperparameterGrid6 = [
    {
        "gamma": [10000.0, 1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001, "auto"],
        "C": [10000.0, 1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001],
        "epsilon": [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
    }
]
BlackBox6 = BlackBox(
    PD.svr_grid_search_predictor, HyperparameterGrid6, str(HyperparameterGrid6)
)

# ------- ANN_grid (BlackBox7)
HyperparameterGrid7 = [
    {
        "hidden_layer_sizes": [
            [1],
            [10],
            [100],
            [1000],
            [1, 1],
            [10, 10],
            [100, 100],
            [1, 10],
            [1, 100],
            [10, 100],
            [100, 10],
            [100, 1],
            [10, 1],
            [1, 1, 1],
            [10, 10, 10],
            [100, 100, 100],
        ]
    }
]
BlackBox7 = BlackBox(
    PD.ann_grid_search_predictor, HyperparameterGrid7, str(HyperparameterGrid7)
)

# ------- GB_grid (BlackBox8)
HyperparameterGrid8 = [
    {
        "n_estimators": [10, 100, 1000],
        "max_depth": [1, 10, 100],
        "learning_rate": [0.01, 0.1, 0.5, 1],
        "loss": ["ls", "lad", "huber", "quantile"],
    }
]  # Learning_rate in range 0 to 1
BlackBox8 = BlackBox(
    PD.gradientboost_gridsearch, HyperparameterGrid8, str(HyperparameterGrid8)
)

# ------- Lasso_grid (BlackBox9)
HyperparameterGrid9 = [
    {
        "alpha": [
            1000000,
            100000,
            10000,
            1000,
            100,
            10,
            1,
            0.1,
            1e-2,
            1e-3,
            1e-4,
            1e-5,
            1e-6,
            1e-7,
            1e-8,
            1e-9,
            1e-10,
        ]
    }
]
BlackBox9 = BlackBox(
    PD.lasso_grid_search_predictor, HyperparameterGrid9, str(HyperparameterGrid9)
)


# ----------------------------------------------- Individual Models --------------------------------------------------

# Splitter functions of Individual Models
def week_weekend_splitter(Dataseries):
    # Datetimeindex format is necessary for individual model methods
    # select all weekday and weekend days from the specific dataseries
    weekday = Dataseries[
        Dataseries.index.dayofweek <= 4
    ]  # here you can change the week / weekend definition
    weekend = Dataseries[Dataseries.index.dayofweek >= 5]
    Dic = {"weekday": weekday, "weekend": weekend}
    return Dic


def hourly_splitter(Dataseries):
    Dic = dict()
    # select all values from each respective hour and add the to a dictionary
    for hour in range(0, 24):
        hourly = Dataseries[Dataseries.index.hour == hour]
        Dic.update({hour: hourly})
    return Dic


class byfeature_splitter:
    'Class for the "byFeature" splitter as with a class the two additional attributes "indivFeature" and "Threshold" can be propagated throughout all following computations'

    def __init__(self, Threshold, Feature, Features_Test, Features_Train="optional"):
        self.Threshold = Threshold
        self.Feature = Feature
        self.Features_Train = Features_Train
        self.Features_Test = Features_Test
        if type(self.Features_Train) != str:
            self.idx_train_above = Features_Train.index[
                Features_Train[self.Feature] >= self.Threshold
            ]
            self.idx_train_below = Features_Train.index[
                Features_Train[self.Feature] < self.Threshold
            ]
        self.idx_test_above = Features_Test.index[
            Features_Test[self.Feature] >= self.Threshold
        ]
        self.idx_test_below = Features_Test.index[
            Features_Test[self.Feature] < self.Threshold
        ]

    def splitter(self, Dataseries):
        # Datetimeindex format is necessary for individual model methods
        # select all above the stated threshold
        if type(self.Features_Train) != str:
            if Dataseries.index.equals(
                self.Features_Train.index
            ):  # check whether Dataseries is within fit or test period
                above = Dataseries.loc[self.idx_train_above]
                below = Dataseries.loc[self.idx_train_below]
        if Dataseries.index.equals(self.Features_Test.index):
            above = Dataseries.loc[self.idx_test_above]
            below = Dataseries.loc[self.idx_test_below]

        Dic = {"above": above, "below": below}
        return Dic


# Individual Models executive functions
class indiv_splitter:
    'Splits the dataframe with the respective "Split_function" in the dataframes needed for training the individual models. The dataframes are safed as needed from the "indiv_model" and "indiv_model_onlypredict" classes'

    def __init__(self, Split_function):
        self.Split_function = Split_function

    def split_train_test(
        self, Features_train, Signal_train, Features_test, Signal_test
    ):
        Dic1 = self.Split_function(Features_train)
        Dic2 = self.Split_function(Signal_train)
        Dic3 = self.Split_function(Features_test)
        Dic4 = self.Split_function(Signal_test)
        Dic = dict()
        for key in Dic1:
            Dic[key] = [Dic1[key], Dic2[key], Dic3[key], Dic4[key]]
        return Dic

    def split_test(self, Features):
        Dic = dict()
        Dic = self.Split_function(Features)
        return Dic

    def split_onlypredict(self, Features_test, Signal_test):
        Dic = dict()
        Dic3 = self.Split_function(Features_test)
        Dic4 = self.Split_function(Signal_test)
        for key in Dic3:
            Dic[key] = [Dic3[key], Dic4[key]]
        return Dic


class indiv_model:
    "Trains the indivdual models and does a prediction"

    def __init__(
        self,
        indiv_splitter_instance,
        Estimator,
        Features_train,
        Signal_train,
        Features_test,
        Signal_test,
        HyperparameterGrid=None,
        CV=None,
        Max_evals=None,
        Recursive=False,
    ):
        self.indiv_splitter_instance = indiv_splitter_instance
        self.Estimator = Estimator
        self.Features_train = Features_train
        self.Signal_train = Signal_train
        self.Features_test = Features_test
        self.Signal_test = Signal_test
        self.HyperparameterGrid = HyperparameterGrid
        self.CV = CV
        self.Max_evals = Max_evals
        self.Recursive = Recursive

    def main(self):
        timestart = time.time()
        Dic = self.indiv_splitter_instance.split_train_test(
            self.Features_train, self.Signal_train, self.Features_test, self.Signal_test
        )

        best_params = dict()
        best_model = dict()

        Y = pd.DataFrame(index=self.Signal_test.index)
        i = 1
        for key in Dic:
            if Dic[key][0].empty:
                Answer = input(
                    "Attention your fit period does not contain data to fit all individual models. An Error is very probable. Proceed anyways?"
                )
                if Answer == "yes" or Answer == "Yes" or Answer == "y" or Answer == "Y":
                    print("Start computing")
                else:
                    sys.exit(
                        "Code stopped by user or invalid user input. Valid is Yes, yes, y and Y."
                    )
            _dic = self.Estimator(
                Features_train=Dic[key][0],
                Signal_train=Dic[key][1],
                Features_test=Dic[key][2],
                Signal_test=Dic[key][3],
                HyperparameterGrid=self.HyperparameterGrid,
                CV=self.CV,
                Max_evals=self.Max_evals,
                Recursive=False,
            )  # fit and infer for the given data #recursive has to be turned of (doesnt work with individual model), it is done later in this function for individual models
            Y_i = _dic["prediction"]  # pull the prediction from the dictionary
            Index = Dic[key][3]
            Y_i = pd.DataFrame(
                index=Index.index, data=Y_i
            )  # reset the index to datetime convention
            Y_i = Y_i.rename(
                columns={0: i}
            )  # rename column per loop to have each period in a single column
            Y = pd.concat([Y, Y_i], axis=1)  # add them all together
            i += 1
            try:  # try to add the best hyperparameters per time intervall, try is necessary since not all estimators pass "best_params"
                best_params[key] = _dic["best_params"]
            except:
                pass
            best_model[key] = _dic["Best_trained_model"]  # add best models to a list

        if self.Recursive == False:
            predicted = Y.sum(
                axis=1
            )  # add all columns together, since each timestamp has only 1 column with a value this is the same as rearranging all the results back to a chronological timeline
        if self.Recursive == True:
            Features_test_i = self.Features_test.copy(deep=True)
            Features_test_i.index = range(
                len(Features_test_i)
            )  # set an trackable index 0,1,2,3,etc.
            Features_test_ii = self.Features_test.copy(deep=True)
            Features_test_ii["TrackIndex"] = range(
                len(self.Features_test)
            )  # add an trackable index to the original one #just for tracking the index of Features_test_i

            # split Features_test_ii into individual model sets
            Dic = self.indiv_splitter_instance.split_test(Features_test_ii)

            for i in Features_test_i.index:
                vector_i = Features_test_i.iloc[
                    [i]
                ]  # get the features of the timestep i

                # if i is in one of the dic[key] data sets, use this key!
                for key in Dic:  # loop through all dictionary entries
                    if not Dic[
                        key
                    ].empty:  # to avoid a crash if not all individual models are called in the test data range
                        if (
                            i in Dic[key].set_index("TrackIndex").index
                        ):  # checks whether the line i is in the data for the data of the respective key
                            OwnLag = best_model[key].infer(
                                vector_i
                            )  # do a one one timestep prediction with the model of the respective key

                Booleans = Features_test_i.columns.str.contains(
                    "_lag_"
                )  # create a Boolean list for with all columns, true for lagged signals, false for other(important: for lagged features it is only "_lag"
                Lagged_column_list = np.array(list(Features_test_i))[Booleans]
                for (
                    columnname
                ) in (
                    Lagged_column_list
                ):  # go through each column containing _lag_ in its name
                    lag = columnname.split("_")[
                        -1
                    ]  # get the lag from the name of the column (lagged signals have the ending, e.g. for lag 1:  "_lag_1"
                    line = (
                        int(lag) + i
                    )  # define the line where the specific prediction should be safed
                    if line < len(
                        Features_test_i
                    ):  # save produced ownlag in features_test_i
                        Features_test_i = Features_test_i.set_value(
                            value=OwnLag,
                            index=line,
                            col=Features_test_i.columns.str.contains("_lag_%s" % lag),
                        )  # set the predicted signal as input for future predictions

            Features_test_i = Features_test_i.set_index(
                self.Signal_test.index
            )  # pd.DataFrame(index=Signal_test.index, data=Features_test_i)  # reset the index to datetime convention
            # split the recursive features_test_i up
            Dic = self.indiv_splitter_instance.split_test(Features_test_i)

            # do an individual model prediction for the "recursive" feature set: Features_test_i
            Y = pd.DataFrame(index=self.Signal_test.index)
            i = 1
            for key in Dic:
                if not Dic[
                    key
                ].empty:  # to avoid a crash if not all individual models are called in the test data range
                    Y_i = best_model[key].infer(Dic[key])
                    Index = Dic[key]
                    Y_i = pd.DataFrame(
                        index=Index.index, data=Y_i
                    )  # reset the index to datetime convention
                    Y_i = Y_i.rename(
                        columns={0: i}
                    )  # rename column per loop to have each period in a single column
                    Y = pd.concat([Y, Y_i], axis=1)  # add them all together
                    i += 1
            predicted = Y.sum(
                axis=1
            )  # add all columns together, since each timestamp has only 1 column with a value this is the same as rearranging all the results back to a chronological timeline

        timeend = time.time()
        return {
            "prediction": predicted,
            "best_params": best_params,
            "ComputationTime": (timeend - timestart),
            "Best_trained_model": best_model,
            "feature_importance": "Not available for individual model",
        }


class indiv_model_onlypredict:
    "Loads a beforehand saved (individual) model and does a prediction"

    def __init__(
        self,
        indiv_splitter_instance,
        Features_test,
        ResultsFolderSubTest,
        NameOfPredictor,
        Recursive,
    ):
        self.indiv_splitter_instance = indiv_splitter_instance
        self.Features_test = Features_test
        self.ResultsFolderSubTest = ResultsFolderSubTest
        self.NameOfPredictor = NameOfPredictor
        self.Recursive = Recursive

    def main(self):
        timestart = time.time()
        Datetimetracker = self.Features_test
        if self.Recursive == False:
            Dic = self.indiv_splitter_instance.split_test(self.Features_test)

            i = 1
            Y = pd.DataFrame(index=self.Features_test.index)
            for key in Dic:
                if not Dic[
                    key
                ].empty:  # to avoid a crash if not all individual models are called in the test data range
                    Predictor = joblib.load(
                        os.path.join(
                            self.ResultsFolderSubTest,
                            "BestModels",
                            "%s_%s.save" % (key, self.NameOfPredictor),
                        )
                    )
                    Y_i = Predictor.infer(Dic[key])  # infer
                    Index = Dic[key]
                    Y_i = pd.DataFrame(
                        index=Index.index, data=Y_i
                    )  # reset the index to datetime convention
                    Y_i = Y_i.rename(
                        columns={0: i}
                    )  # rename column per loop to have each period in a single column
                    Y = pd.concat([Y, Y_i], axis=1)  # add them all together
                    i += 1
            predicted = Y.sum(
                axis=1
            )  # add all columns together, since each timestamp has only 1 column with a value this is the same as rearranging all the results back to a chronological timeline
        if self.Recursive == True:
            Features_test_i = self.Features_test.copy(deep=True)
            Features_test_i.index = range(
                len(Features_test_i)
            )  # set an trackable index 0,1,2,3,etc.
            Features_test_ii = self.Features_test.copy(deep=True)
            Features_test_ii["TrackIndex"] = range(
                len(self.Features_test)
            )  # add an trackable index to the original one #just for tracking the index of Features_test_i

            # split Features_test_ii into individual model sets
            Dic = self.indiv_splitter_instance.split_test(Features_test_ii)

            for i in Features_test_i.index:
                vector_i = Features_test_i.iloc[
                    [i]
                ]  # get the features of the timestep i

                # if i is in one of the dic[key] data sets, use this key!
                for key in Dic:  # loop through all dictionary entries
                    if not Dic[
                        key
                    ].empty:  # to avoid a crash if not all individual models are called in the test data range
                        if (
                            i in Dic[key].set_index("TrackIndex").index
                        ):  # checks whether the line i is in the data for the data of the respective key
                            Predictor = joblib.load(
                                os.path.join(
                                    self.ResultsFolderSubTest,
                                    "BestModels",
                                    "%s_%s.save" % (key, self.NameOfPredictor),
                                )
                            )  # load the respective model
                            OwnLag = Predictor.infer(
                                vector_i
                            )  # do a one one timestep prediction with the model of the respective key

                Booleans = Features_test_i.columns.str.contains(
                    "_lag_"
                )  # create a Boolean list for with all columns, true for lagged signals, false for other(important: for lagged features it is only "_lag"
                Lagged_column_list = np.array(list(Features_test_i))[Booleans]
                for (
                    columnname
                ) in (
                    Lagged_column_list
                ):  # go through each column containing _lag_ in its name
                    lag = columnname.split("_")[
                        -1
                    ]  # get the lag from the name of the column (lagged signals have the ending, e.g. for lag 1:  "_lag_1"
                    line = (
                        int(lag) + i
                    )  # define the line where the specific prediction should be safed
                    if line < len(
                        Features_test_i
                    ):  # save produced ownlag in features_test_i
                        Features_test_i = Features_test_i.set_value(
                            value=OwnLag,
                            index=line,
                            col=Features_test_i.columns.str.contains("_lag_%s" % lag),
                        )  # set the predicted signal as input for future predictions

            Features_test_i = Features_test_i.set_index(
                Datetimetracker.index
            )  # pd.DataFrame(index=Signal_test.index, data=Features_test_i)  # reset the index to datetime convention

            # split the recursive features_test_i up
            Dic = self.indiv_splitter_instance.split_test(Features_test_i)

            # do an individual model prediction for the "recursive" feature set: Features_test_i
            Y = pd.DataFrame(index=Datetimetracker.index)
            i = 1
            for key in Dic:
                if not Dic[
                    key
                ].empty:  # to avoid a crash if not all individual models are called in the test data range
                    Predictor = joblib.load(
                        os.path.join(
                            self.ResultsFolderSubTest,
                            "BestModels",
                            "%s_%s.save" % (key, self.NameOfPredictor),
                        )
                    )  # load the respective model
                    Y_i = Predictor.infer(Dic[key])
                    Index = Dic[key]
                    Y_i = pd.DataFrame(
                        index=Index.index, data=Y_i
                    )  # reset the index to datetime convention
                    Y_i = Y_i.rename(
                        columns={0: i}
                    )  # rename column per loop to have each period in a single column
                    Y = pd.concat([Y, Y_i], axis=1)  # add them all together
                    i += 1
            predicted = Y.sum(
                axis=1
            )  # add all columns together, since each timestamp has only 1 column with a value this is the same as rearranging all the results back to a chronological timeline
        return predicted


# End of Individual Models
# -----------------------------------------------------------------------------------------------------------------------

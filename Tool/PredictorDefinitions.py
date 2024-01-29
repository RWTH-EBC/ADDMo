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

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
from hyperopt.pyll import scope
import hyperopt.pyll.stochastic
from sklearn.externals import joblib
import os
import warnings

from ModelTuningRuntimeResults import *

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# a recursive plugin which can be used in every BB Model in order to create a recursive behavior
def recursive(Features_test, Best_trained_model):
    Features_test_i = Features_test.copy(deep=True)
    Features_test_i.index = range(
        len(Features_test_i)
    )  # set an trackable index 0,1,2,3,etc.
    for i in Features_test_i.index:
        vector_i = Features_test_i.iloc[[i]]  # get the features of the timestep i
        OwnLag = Best_trained_model.predict(
            vector_i
        )  # do a one one timestep prediction
        Booleans = Features_test_i.columns.str.contains(
            "_lag_"
        )  # create a Boolean list for with all columns, true for lagged signals, false for other(important: for lagged features it is only "_lag"
        Lagged_column_list = np.array(list(Features_test_i))[Booleans]
        for (
            columnname
        ) in Lagged_column_list:  # go through each column containing _lag_ in its name
            lag = columnname.split("_")[
                -1
            ]  # get the lag from the name of the column (lagged signals have the ending, e.g. for lag 1:  "_lag_1"
            line = (
                int(lag) + i
            )  # define the line where the specific prediction should be safed
            if line < len(Features_test_i):
                Features_test_i = Features_test_i.set_value(
                    value=OwnLag,
                    index=line,
                    col=Features_test_i.columns.str.contains("_lag_%s" % lag),
                )  # set the predicted signal as input for future predictions
    return Features_test_i


# -------------------------------------- Predictor definitions ------------------------------------------------------
# Follow the following 4 steps when adding a new predictor:

# 1. Add the names of the new predictors to 'AvailablePredictors' and also in the docstrings below
"""
Name of the predictors used:

SVR = SVR Bayesian
RF = RF predictor
ANN = ANN Bayesian
GB = Gradient Boost Bayesian
Lasso = Lasso Bayesian
SVR_grid = SVR Grid Search
ANN_grid = ANN Grid Search
GB_grid = Gradient Boost Grid Search
Lasso_grid = Lasso Grid Search
"""
AvailablePredictors = [
    "SVR",
    "RF",
    "ANN",
    "GB",
    "Lasso",
    "SVR_grid",
    "ANN_grid",
    "GB_grid",
    "Lasso_grid",
]


# 2. Add the predictor summary as a class attribute to ModelTuningRuntimeResults class in ModelTuningRuntimeResults.py
# Use the below format to do so:
"""
ModelTuningRuntimeResults is a class that stores all the statistical runtime info about each predictor, hence 
when a new predictor is defined it must added as a class attribute in the format shown below:

Format: self.'name_of_the_predictor'_Summary = RRSummary()

Example:
self.Lasso_Summary = RRSummary() #if the predictor added is Lasso

"""
# 3. Add the predictor summary to the list below


def get_model_summary_object_list(MT_RR_object):
    ModelSummaryObjectList = [
        MT_RR_object.SVR_Summary,
        MT_RR_object.RF_Summary,
        MT_RR_object.ANN_Summary,
        MT_RR_object.GB_Summary,
        MT_RR_object.Lasso_Summary,
        MT_RR_object.SVR_grid_Summary,
        MT_RR_object.ANN_grid_Summary,
        MT_RR_object.GB_grid_Summary,
        MT_RR_object.Lasso_grid_Summary,
    ]
    return ModelSummaryObjectList


# 4. Define the predictor's internal workings
def svr_grid_search_predictor(
    Features_train,
    Signal_train,
    Features_test,
    Signal_test,
    HyperparameterGrid,
    CV,
    Max_evals=NotImplemented,
    Recursive=False,
):
    # print("Cell GridSearchSVR start---------------------------------------------------------")
    timestart = time.time()

    # HyperparameterGrid= [{'gamma': [1e4 , 1 , 1e-4, 'auto'],'C': [1e-4, 1, 1e4],'epsilon': [1, 1e-4]}]

    Signal_test = Signal_test.values.ravel()
    # Features_test = Features_test.values.ravel() #this one not in order to have recursive still working fine
    Signal_train = Signal_train.values.ravel()
    Features_train = Features_train.values

    # gridsearch through
    svr = GridSearchCV(SVR(cache_size=1500), HyperparameterGrid, cv=CV, scoring="r2")
    Best_trained_model = svr.fit(Features_train, Signal_train)
    if not Features_test.empty:
        if Recursive == False:
            predicted = Best_trained_model.predict(Features_test)
        elif Recursive == True:
            Features_test_i = recursive(Features_test, Best_trained_model)
            predicted = Best_trained_model.predict(Features_test_i)
        score = Best_trained_model.score(Features_test, Signal_test)
    else:
        predicted = []
        score = "empty"

    # print section
    # print("The Score svr: %s" %Best_trained_model.score_test(Features_test, Signal_test))
    # print("Best Hyperparameters: %s" %svr.best_params_)
    timeend = time.time()
    # print("SVR took %s seconds" %(timeend-timestart))
    return {
        "score_test": score,
        "best_params": svr.best_params_,
        "prediction": predicted,
        "ComputationTime": (timeend - timestart),
        "Best_trained_model": Best_trained_model,
        "feature_importance": "Not available for that model",
    }


def svr_bayesian_predictor(
    Features_train,
    Signal_train,
    Features_test,
    Signal_test,
    HyperparameterGrid,
    CV,
    Max_evals,
    Recursive=False,
):
    # print("Cell Bayesian Optimization SVR start---------------------------------------------------------")
    timestart = time.time()

    # HyperparameterGrid = {"C": hp.loguniform("C", -6, 23.025), "gamma":hp.loguniform("gamma", -6,23.025), "epsilon":hp.loguniform("epsilon", -6, 23.025)} #if using loguniform, e.g. you want the parameter range 0.1 to 1000 type in (log(0.1), log(1000))

    Signal_train = Signal_train.values.ravel()
    Features_train = Features_train.values

    def hyperopt_cv(params):
        t_start = time.time()
        Estimator = SVR(
            **params, cache_size=1500
        )  # give the specific parameter sample per run from fmin
        CV_score = cross_val_score(
            estimator=Estimator, X=Features_train, y=Signal_train, cv=CV, scoring="r2"
        ).mean()  # create a crossvalidation score_test which shall be optimized
        t_end = time.time()
        print(
            "Params per iteration: %s \ with the cross-validation score_test %.3f, took %.2fseconds"
            % (params, CV_score, (t_end - t_start))
        )
        return CV_score

    def f(params):
        acc = hyperopt_cv(params)
        return {
            "loss": -acc,
            "status": STATUS_OK,
        }  # fmin always minimizes the loss function, we want acc to maximize-> (-acc)

    trials = Trials()  # this is for tracking the bayesian optimization
    BestParams = fmin(
        f, HyperparameterGrid, algo=tpe.suggest, max_evals=Max_evals, trials=trials
    )  # do the bayesian optimization
    Best_trained_model = SVR(**BestParams).fit(
        Features_train, Signal_train
    )  # set the best hyperparameter to the SVR machine
    if not Features_test.empty:
        if Recursive == False:
            predicted = Best_trained_model.predict(Features_test)
        elif Recursive == True:
            Features_test_i = recursive(Features_test, Best_trained_model)
            predicted = Best_trained_model.predict(Features_test_i)
        score = Best_trained_model.score(Features_test, Signal_test)
    else:
        predicted = []
        score = "empty"

    # print section
    # print("Bayesian Optimization Parameters")
    # print("Everything about the search: %s" %trials.trials)
    # print("List of returns of \"Objective\": %s" %trials.results)
    # print("List of losses per ok trial: %s" %trials.losses())
    # print("List of statuses: %s" %trials.statuses())
    # print("BlackBox Parameter")
    # print("The Score svr: %s" %Best_trained_model.score_test(Features_test, Signal_test))
    # print("Best Hyperparameters: %s" %BestParams)
    timeend = time.time()
    # print("SVR took %s seconds" %(timeend-timestart))
    return {
        "score_test": score,
        "best_params": BestParams,
        "prediction": predicted,
        "ComputationTime": (timeend - timestart),
        "Best_trained_model": Best_trained_model,
        "feature_importance": "Not available for that model",
    }


def rf_predictor(
    Features_train,
    Signal_train,
    Features_test,
    Signal_test,
    HyperparameterGrid=NotImplemented,
    CV=NotImplemented,
    Max_evals=NotImplemented,
    Recursive=False,
):
    # print("Cell RandomForest start---------------------------------------------------------")
    timestart = time.time()

    Signal_test = Signal_test.values.ravel()
    # Features_test = RandomForestRegressor
    # Features_test.values.ravel() #this one not in order to have recursive still working fine
    Signal_train = Signal_train.values.ravel()
    Features_train = Features_train.values

    # using RandomForest
    rf = RandomForestRegressor()  # here you could state a max_depth for rf
    Best_trained_model = rf.fit(Features_train, Signal_train)
    if (
        not Features_test.empty
    ):  # check whether the test data is not empty #todo:finish(seems to work now but still do for the other models) (maybe better as a class, think an plan)(didnt work because there is no Signal_test for scoring or doing the score_test; check whether score_test is necessary anyways, because it is scored later on)(Think of objectoriented programming, there you could apply the function "score_test" inside the class and only call it if necesaarry
        if Recursive == False:
            predicted = Best_trained_model.predict(Features_test)
        elif Recursive == True:
            Features_test_i = recursive(Features_test, Best_trained_model)
            predicted = Best_trained_model.predict(Features_test_i)
        score = Best_trained_model.score(Features_test, Signal_test)
    else:
        predicted = []
        score = "empty"

    # print section
    print("The Score rf: %s" % score)
    # print("Feature Importance RF: %s" %Best_trained_model.feature_importances_)
    timeend = time.time()
    print("RF took %s seconds" % (timeend - timestart))
    return {
        "score_test": score,
        "feature_importance": Best_trained_model.feature_importances_,
        "prediction": predicted,
        "ComputationTime": (timeend - timestart),
        "Best_trained_model": Best_trained_model,
        "best_params": "Not available for RF",
    }


def gradientboost_gridsearch(
    Features_train,
    Signal_train,
    Features_test,
    Signal_test,
    HyperparameterGrid,
    CV,
    Max_evals=NotImplemented,
    Recursive=False,
):
    # print("Cell GradientBoost start---------------------------------------------------------")
    timestart = time.time()

    # HyperparameterGrid = [{"n_estimators" : [10,100,1000,10000,100000], "max_depth" : [0.1,1,10,100,1000], "learning_rate" : [0.01,0.1,0.5,1], "loss" : ["ls", "lad", "huber", "quantile"]}]

    # using gradient boosting with gridsearch
    gb = GridSearchCV(
        GradientBoostingRegressor(), HyperparameterGrid, cv=CV, scoring="r2"
    )
    gb = gb.fit(Features_train, Signal_train)

    # A single gb with the paramaters found in Gridsearch is implemented in order to be able to use the .feature_importances_ attribute and see the influence of the features
    bestgb = GradientBoostingRegressor()
    bestgb = bestgb.set_params(**gb.best_params_)
    Best_trained_model = bestgb.fit(Features_train, Signal_train)
    if not Features_test.empty:
        if Recursive == False:
            predicted = Best_trained_model.predict(Features_test)
        elif Recursive == True:
            Features_test_i = recursive(Features_test, Best_trained_model)
            predicted = Best_trained_model.predict(Features_test_i)
        score = Best_trained_model.score_test(Features_test, Signal_test)
    else:
        predicted = []
        score = "empty"

    # print section
    # print("The Score gb: %s" %Best_trained_model.score_test(Features_test, Signal_test))
    # print("Feature Importance gb: %s" %Best_trained_model.feature_importances_)
    # print("best_params: %s" %gb.best_params_)
    timeend = time.time()
    # print("gb took %s seconds" %(timeend-timestart))
    return {
        "score_test": score,
        "best_params": gb.best_params_,
        "feature_importance": Best_trained_model.feature_importances_,
        "prediction": predicted,
        "ComputationTime": (timeend - timestart),
        "Best_trained_model": Best_trained_model,
    }


def gradientboost_bayesian(
    Features_train,
    Signal_train,
    Features_test,
    Signal_test,
    HyperparameterGrid,
    CV,
    Max_evals,
    Recursive=False,
):
    # print("Cell Bayesian Optimization GB start---------------------------------------------------------")
    timestart = time.time()

    # HyperparameterGrid = {"n_estimators": scope.int(hp.qloguniform("n_estimators", log(1), log(1e3), 1)),
    #                      "max_depth": scope.int(hp.qloguniform("max_depth", log(1), log(100), 1)),
    #                      "learning_rate": hp.loguniform("learning_rate", log(1e-2), log(1)), "loss": hp.choice("loss",
    #                                                                                                            ["ls",
    #                                                                                                             "lad",
    #                                                                                                             "huber",
    #                                                                                                             "quantile"])}  # if anything except numbers is changed, please change the respective code lines for converting notation style in the gradienboost_bayesian function

    Signal_test = Signal_test.values.ravel()
    # Features_test = Features_test.values.ravel() #this one not in order to have recursive still working fine
    Signal_train = Signal_train.values.ravel()
    Features_train = Features_train.values

    def hyperopt_cv(params):
        t_start = time.time()
        Estimator = GradientBoostingRegressor(
            **params
        )  # give the specific parameter sample per run from fmin
        CV_score = cross_val_score(
            estimator=Estimator, X=Features_train, y=Signal_train, cv=CV, scoring="r2"
        ).mean()  # create a crossvalidation score_test which shall be optimized
        t_end = time.time()
        print(
            "Params per iteration: %s \ with the cross-validation score_test %.3f, took %.2fseconds"
            % (params, CV_score, (t_end - t_start))
        )
        return CV_score

    def f(params):
        acc = hyperopt_cv(params)
        return {
            "loss": -acc,
            "status": STATUS_OK,
        }  # fmin always minimizes the loss function, we want acc to maximize-> (-acc)

    trials = Trials()  # this is for tracking the bayesian optimization
    BestParams = fmin(
        f, HyperparameterGrid, algo=tpe.suggest, max_evals=Max_evals, trials=trials
    )  # do the bayesian optimization

    # converting notation style
    max_depth = int(BestParams["max_depth"])
    n_estimators = int(BestParams["n_estimators"])
    learning_rate = BestParams["learning_rate"]
    loss = ["ls", "lad", "huber", "quantile"][BestParams["loss"]]
    BestParams = {
        "learning_rate": learning_rate,
        "loss": loss,
        "max_depth": max_depth,
        "n_estimators": n_estimators,
    }

    Best_trained_model = GradientBoostingRegressor(**BestParams).fit(
        Features_train, Signal_train
    )  # set the best hyperparameter to the SVR machine
    if not Features_test.empty:
        if Recursive == False:
            predicted = Best_trained_model.predict(Features_test)
        elif Recursive == True:
            Features_test_i = recursive(Features_test, Best_trained_model)
            predicted = Best_trained_model.predict(Features_test_i)
        score = Best_trained_model.score(Features_test, Signal_test)
    else:
        predicted = []
        score = "empty"

    # print section
    # print("Bayesian Optimization Parameters")
    # print("Everything about the search: %s" %trials.trials)
    # print("List of returns of \"Objective\": %s" %trials.results)
    # print("List of losses per ok trial: %s" %trials.losses())
    # print("List of statuses: %s" %trials.statuses())
    # print("BlackBox Parameter")
    # print("The Score GB: %s" %Best_trained_model.score_test(Features_test, Signal_test))
    # print("Best Hyperparameters: %s" %BestParams)
    timeend = time.time()
    # print("GB took %s seconds" %(timeend-timestart))
    return {
        "score_test": score,
        "feature_importance": Best_trained_model.feature_importances_,
        "best_params": BestParams,
        "prediction": predicted,
        "ComputationTime": (timeend - timestart),
        "Best_trained_model": Best_trained_model,
    }


def lasso_grid_search_predictor(
    Features_train,
    Signal_train,
    Features_test,
    Signal_test,
    HyperparameterGrid,
    CV,
    Max_evals=NotImplemented,
    Recursive=False,
):
    # print("Cell Lasso start----------------------------------------------------------------")
    timestart = time.time()

    # HyperparameterGrid=[{'alpha':[100000, 10, 1, 0.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]}]

    # gridsearch Lasso
    lasso = GridSearchCV(
        linear_model.Lasso(max_iter=1000000), HyperparameterGrid, cv=CV
    )
    lasso = lasso.fit(Features_train, Signal_train)

    # A single Lasso with the paramaters found in Gridsearch is implemented in order to be able to use the .coef_ attribute and see the influence of the features
    bestlasso = linear_model.Lasso(max_iter=1000000)
    bestlasso = bestlasso.set_params(**lasso.best_params_)
    Best_trained_model = bestlasso.fit(Features_train, Signal_train)
    if not Features_test.empty:
        if Recursive == False:
            predicted = Best_trained_model.predict(Features_test)
        elif Recursive == True:
            Features_test_i = recursive(Features_test, Best_trained_model)
            predicted = Best_trained_model.predict(Features_test_i)
        score = Best_trained_model.score_test(Features_test, Signal_test)
    else:
        predicted = []
        score = "empty"

    # print section
    timeend = time.time()
    # print("The Score Lasso: %s" % Best_trained_model.score_test(Features_test, Signal_test))
    # print("Best Hyperparameters: %s" %lasso.best_params_)
    # print("Lasso coef: %s" % Best_trained_model.coef_)
    # print("Lasso took %s seconds" %(timeend-timestart))

    return {
        "score_test": score,
        "best_params": lasso.best_params_,
        "feature_importance": Best_trained_model.coef_,
        "prediction": predicted,
        "ComputationTime": (timeend - timestart),
        "Best_trained_model": Best_trained_model,
    }


def lasso_bayesian(
    Features_train,
    Signal_train,
    Features_test,
    Signal_test,
    HyperparameterGrid,
    CV,
    Max_evals,
    Recursive=False,
):
    # print("Cell Bayesian Optimization Lasso start---------------------------------------------------------")
    timestart = time.time()

    # HyperparameterGrid = {"alpha": hp.loguniform("alpha", log(1e-10), log(1e6))}
    Signal_test = Signal_test.values.ravel()
    # Features_test = Features_test.values.ravel() #this one not in order to have recursive still working fine
    Signal_train = Signal_train.values.ravel()
    Features_train = Features_train.values

    def hyperopt_cv(params):
        t_start = time.time()
        Estimator = linear_model.Lasso(
            **params, max_iter=1000000
        )  # give the specific parameter sample per run from fmin
        CV_score = cross_val_score(
            estimator=Estimator, X=Features_train, y=Signal_train, cv=CV, scoring="r2"
        ).mean()  # create a crossvalidation score_test which shall be optimized
        t_end = time.time()
        print(
            "Params per iteration: %s \ with the cross-validation score_test %.3f, took %.2fseconds"
            % (params, CV_score, (t_end - t_start))
        )
        return CV_score

    def f(params):
        acc = hyperopt_cv(params)
        return {
            "loss": -acc,
            "status": STATUS_OK,
        }  # fmin always minimizes the loss function, we want acc to maximize-> (-acc)

    trials = Trials()  # this is for tracking the bayesian optimization
    BestParams = fmin(
        f, HyperparameterGrid, algo=tpe.suggest, max_evals=Max_evals, trials=trials
    )  # do the bayesian optimization
    Best_trained_model = linear_model.Lasso(**BestParams, max_iter=1000000).fit(
        Features_train, Signal_train
    )  # set the best hyperparameter to the SVR machine
    if not Features_test.empty:
        if Recursive == False:
            predicted = Best_trained_model.predict(Features_test)
        elif Recursive == True:
            Features_test_i = recursive(Features_test, Best_trained_model)
            predicted = Best_trained_model.predict(Features_test_i)
        score = Best_trained_model.score(Features_test, Signal_test)
    else:
        predicted = []
        score = "empty"

    # print section
    # print("Bayesian Optimization Parameters")
    # print("Everything about the search: %s" %trials.trials)
    # print("List of returns of \"Objective\": %s" %trials.results)
    # print("List of losses per ok trial: %s" %trials.losses())
    # print("List of statuses: %s" %trials.statuses())
    # print("BlackBox Parameter")
    # print("The Score Lasso: %s" %Best_trained_model.score_test(Features_test, Signal_test))
    # print("Best Hyperparameters: %s" %BestParams)
    timeend = time.time()
    # print("Lasso took %s seconds" %(timeend-timestart))
    return {
        "score_test": score,
        "feature_importance": Best_trained_model.coef_,
        "best_params": BestParams,
        "prediction": predicted,
        "ComputationTime": (timeend - timestart),
        "Best_trained_model": Best_trained_model,
    }


def ann_grid_search_predictor(
    Features_train,
    Signal_train,
    Features_test,
    Signal_test,
    HyperparameterGrid,
    CV,
    Max_evals=NotImplemented,
    Recursive=False,
):
    # print("Cell GridSearchANN start---------------------------------------------------------")
    timestart = time.time()

    # HyperparameterGrid= [{'hidden_layer_sizes':[[1],[10],[100],[1000],[1, 1],[10, 10], [100, 100],[1,10],[1,100],[10,100],[100,10],[100,1],[10,1],[1, 1, 1],[10, 10, 10],[100,100,100]]}]

    # gridsearch with MLP
    ann = GridSearchCV(MLPRegressor(max_iter=1000000), HyperparameterGrid, cv=CV)
    Best_trained_model = ann.fit(Features_train, Signal_train)
    if not Features_test.empty:
        if Recursive == False:
            predicted = Best_trained_model.predict(Features_test)
        elif Recursive == True:
            Features_test_i = recursive(Features_test, Best_trained_model)
            predicted = Best_trained_model.predict(Features_test_i)
        score = Best_trained_model.score(Features_test, Signal_test)
    else:
        predicted = []
        score = "empty"

    timeend = time.time()
    # print section
    # print("The Score ann: %s" %Best_trained_model.score_test(Features_test, Signal_test))
    # print("Best Hyperparameters: %s" %ann.best_params_)
    # print("ANN took %s seconds" %(timeend-timestart))

    return {
        "score_test": score,
        "best_params": ann.best_params_,
        "prediction": predicted,
        "ComputationTime": (timeend - timestart),
        "Best_trained_model": Best_trained_model,
        "feature_importance": "Not available for that model",
    }


def ann_bayesian_predictor(
    Features_train,
    Signal_train,
    Features_test,
    Signal_test,
    HyperparameterGrid,
    CV,
    Max_evals,
    Recursive=False,
):
    # print("Cell Bayesian Optimization ANN start---------------------------------------------------------")
    timestart = time.time()

    # HyperparameterGrid= hp.choice("number_of_layers",
    #                    [
    #                    {"1layer": scope.int(hp.qloguniform("1.1", log(1), log(1000), 1))},
    #                    {"2layer": [scope.int(hp.qloguniform("1.2", log(1), log(1000), 1)), scope.int(hp.qloguniform("2.2", log(1), log(1000), 1))]},
    #                    {"3layer": [scope.int(hp.qloguniform("1.3", log(1), log(1000), 1)), scope.int(hp.qloguniform("2.3", log(1), log(1000), 1)), scope.int(hp.qloguniform("3.3", log(1), log(1000), 1))]}
    #                    ])

    Signal_test = Signal_test.values.ravel()
    # Features_test = Features_test.values.ravel() #this one not in order to have recursive still working fine
    Signal_train = Signal_train.values.ravel()
    Features_train = Features_train.values

    def hyperopt_cv(params):
        t_start = time.time()
        try:  # set params so that it fits the estimators attribute style
            params = {"hidden_layer_sizes": params["1layer"]}
        except:
            try:
                params = {"hidden_layer_sizes": params["2layer"]}
            except:
                try:
                    params = {"hidden_layer_sizes": params["3layer"]}
                except:
                    sys.exit(
                        "Your bayesian hyperparametergrid does not fit the requirements, check the example and/or change the hyperparametergrid or the postprocessing in def hyperopt_cv"
                    )
        Estimator = MLPRegressor(
            **params, max_iter=1000000
        )  # give the specific parameter sample per run from fmin
        CV_score = cross_val_score(
            estimator=Estimator, X=Features_train, y=Signal_train, cv=CV, scoring="r2"
        ).mean()  # create a crossvalidation score_test which shall be optimized
        t_end = time.time()
        print(
            "Params per iteration: %s \ with the cross-validation score_test %.3f, took %.2fseconds"
            % (params, CV_score, (t_end - t_start))
        )
        return CV_score

    def f(params):
        acc = hyperopt_cv(params)
        return {
            "loss": -acc,
            "status": STATUS_OK,
        }  # fmin always minimizes the loss function, we want acc to maximize-> (-acc)

    trials = Trials()
    BestParams = fmin(
        f, HyperparameterGrid, algo=tpe.suggest, max_evals=Max_evals, trials=trials
    )
    try:  # set params so that it fits the estimators attribute style
        Z = [int(BestParams["1.1"])]
    except:
        try:
            Z = [int(BestParams["1.2"]), int(BestParams["2.2"])]
        except:
            try:
                Z = [
                    int(BestParams["1.3"]),
                    int(BestParams["2.3"]),
                    int(BestParams["2.3"]),
                ]
            except:
                sys.exit(
                    "Your bayesian hyperparametergrid does not fit the requirements, check the example and/or change the hyperparametergrid or the postprocessing for the bestparams in ann_bayesian_predictor"
                )
    BestParams = {
        "hidden_layer_sizes": Z
    }  # set params so that it fits the estimators attribute style
    Ann_best = MLPRegressor(
        **BestParams
    )  # set the best hyperparameter to the SVR machine
    Best_trained_model = Ann_best.fit(Features_train, Signal_train)
    if not Features_test.empty:
        if Recursive == False:
            predicted = Best_trained_model.predict(Features_test)
        elif Recursive == True:
            Features_test_i = recursive(Features_test, Best_trained_model)
            predicted = Best_trained_model.predict(Features_test_i)
        score = Ann_best.score(
            Features_test, Signal_test
        )  # Todo: Ann_best or should it be Best_trained_model?
    else:
        predicted = []
        score = "empty"

    # print section
    # print("Bayesian Optimization Parameters")
    # print("Everything about the search: %s" %trials.trials)
    # print("List of returns of \"Objective\": %s" %trials.results)
    # print("List of losses per ok trial: %s" %trials.losses())
    # print("List of statuses: %s" %trials.statuses())
    # print("BlackBox Parameter")
    # print("The Score ann: %s" %Ann_best.score_test(Features_test, Signal_test))
    # print("Best Hyperparameters: %s" %BestParams)
    timeend = time.time()
    # print("ANN took %s seconds" %(timeend-timestart))
    return {
        "score_test": score,
        "best_params": BestParams,
        "prediction": predicted,
        "ComputationTime": (timeend - timestart),
        "Best_trained_model": Best_trained_model,
        "feature_importance": "Not available for that model",
    }


# End of Predictor Definitions
# -----------------------------------------------------------------------------------------------------------------------

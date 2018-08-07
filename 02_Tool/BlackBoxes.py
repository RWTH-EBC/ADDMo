from __future__ import print_function
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


########################################################################################################################
#Hyperparameter Tuning Information:
#Define a Train/Test area where Crossvalidation shall be performed, define also a predict area
#where an additional prediction is evaluated with the in the Train/Test area found hyperparameters and training wheights.
#The model with the tuned hyperparameter is after all trained on the whole Train/Test data set.

#See below each function a example for how to pull which result from the function (A dictionary is used as return value)

#For exemplary HyperparameterGrid see each black-box models comment
########################################################################################################################
'''#Trial Blackboxes with classes (OOP)
class Estimator():
    def __init__(self, Estimator,):
        self.Estimator = Estimator

    def predicting(self):
        if Recursive == False:
            Predicted = Best_trained_model.predict(Features_test)
        elif Recursive == True:
            Features_test_i = recursive(Features_test, Best_trained_model)
            Predicted = Best_trained_model.predict(Features_test_i)
        return Predicted


class grid_search(Estimator):
    def __init__(self):

    def training(self):
        Signal_train = Signal_train.values.ravel()
        Features_train = Features_train.values
        est = GridSearchCV(Estimator, HyperparameterGrid, cv=CV_DT, scoring="r2")
        Best_trained_model = est.fit(Features_train, Signal_train)
        return Best_trained_model

class bayesian_SVR(Estimator):
'''
'''#Trial BlackBoxes -------------------------------------------------------------------------
def predicting(TrainedModel, Features_test, Signal_test, Recursive=False):
    if Recursive == False:
        predicted = TrainedModel.predict(Features_test)
    elif Recursive == True:
        Features_test_i = recursive(Features_test, TrainedModel)
        predicted = TrainedModel.predict(Features_test_i)

    return {"score" : Best_trained_model.score(Features_test, Signal_test),
            "prediction" : predicted}

def svr_bayesian_predictor(Features_train, Signal_train, HyperparameterGrid, CV_DT, Max_evals, ScoreFunc="r2"):
    #print("Cell Bayesian Optimization SVR start---------------------------------------------------------")
    timestart = time.time()

    #HyperparameterGrid = {"C": hp.loguniform("C", -6, 23.025), "gamma":hp.loguniform("gamma", -6,23.025), "epsilon":hp.loguniform("epsilon", -6, 23.025)} #if using loguniform, e.g. you want the parameter range 0.1 to 1000 type in (log(0.1), log(1000))

    #For faster training:
    Signal_train = Signal_train.values.ravel()
    Features_train = Features_train.values

    def hyperopt_cv(params):
        t_start = time.time()
        Estimator = SVR(**params, cache_size=1500)    #give the specific parameter sample per run from fmin
        CV_score = cross_val_score(estimator=Estimator, X=Features_train, y=Signal_train, cv=CV_DT, scoring=ScoreFunc).mean()  # create a crossvalidation score which shall be optimized
        t_end = time.time()
        print("Params per iteration: %s \ with the cross-validation score %.3f, took %.2fseconds" % (params, CV_score, (t_end-t_start)))
        return CV_score

    def f(params):
        acc = hyperopt_cv(params)
        return {"loss": -acc, "status": STATUS_OK} #fmin always minimizes the loss function, we want acc to maximize-> (-acc)

    trials = Trials() #this is for tracking the bayesian optimization
    BestParams = fmin(f, HyperparameterGrid, algo=tpe.suggest, max_evals=Max_evals, trials=trials) #do the bayesian optimization
    Best_trained_model = SVR(**BestParams).fit(Features_train, Signal_train)    #set the best hyperparameter to the SVR machine

    #print section
    #print("Bayesian Optimization Parameters")
    #print("Everything about the search: %s" %trials.trials)
    #print("List of returns of \"Objective\": %s" %trials.results)
    #print("List of losses per ok trial: %s" %trials.losses())
    #print("List of statuses: %s" %trials.statuses())
    #print("BlackBox Parameter")
    #print("The Score svr: %s" %Best_trained_model.score(Features_test, Signal_test))
    #print("Best Hyperparameters: %s" %BestParams)
    timeend = time.time()
    #print("SVR took %s seconds" %(timeend-timestart))
    return {"best_params" : BestParams,
            "ComputationTime" : (timeend-timestart),
            "Best_trained_model": Best_trained_model}

#Trial end---------------------------------------------------------------------------------'''



#a recursive plugin which can be used in every BB Model in order to create a recursive behavior
def recursive(Features_test, Best_trained_model):
    Features_test_i = Features_test.copy(deep=True)
    Features_test_i.index = range(len(Features_test_i))  # set an trackable index 0,1,2,3,etc.
    for i in Features_test_i.index:
        vector_i = Features_test_i.iloc[[i]] #get the features of the timestep i
        OwnLag = Best_trained_model.predict(vector_i) #do a one one timestep prediction
        Booleans = Features_test_i.columns.str.contains("_lag_") #create a Boolean list for with all columns, true for lagged signals, false for other(important: for lagged features it is only "_lag"
        Lagged_column_list = np.array(list(Features_test_i))[Booleans]
        for columnname in Lagged_column_list:  # go through each column containing _lag_ in its name
            lag = columnname.split("_")[-1]  # get the lag from the name of the column (lagged signals have the ending, e.g. for lag 1:  "_lag_1"
            line = int(lag) + i  # define the line where the specific prediction should be safed
            if line < len(Features_test_i):
                Features_test_i = Features_test_i.set_value(value=OwnLag, index=line, col=Features_test_i.columns.str.contains("_lag_%s" % lag)) #set the predicted signal as input for future predictions
    return Features_test_i

def svr_grid_search_predictor(Features_train, Signal_train, Features_test, Signal_test, HyperparameterGrid, CV, Max_evals=NotImplemented, Recursive=False):
    #print("Cell GridSearchSVR start---------------------------------------------------------")
    timestart = time.time()

    #HyperparameterGrid= [{'gamma': [1e4 , 1 , 1e-4, 'auto'],'C': [1e-4, 1, 1e4],'epsilon': [1, 1e-4]}]

    Signal_test = Signal_test.values.ravel()
    #Features_test = Features_test.values.ravel() #this one not in order to have recursive still working fine
    Signal_train = Signal_train.values.ravel()
    Features_train = Features_train.values


    #gridsearch through
    svr = GridSearchCV(SVR(cache_size = 1500), HyperparameterGrid, cv=CV, scoring="r2")
    Best_trained_model = svr.fit(Features_train, Signal_train)
    if Recursive == False:
        predicted = Best_trained_model.predict(Features_test)
    elif Recursive == True:
        Features_test_i = recursive(Features_test, Best_trained_model)
        predicted = Best_trained_model.predict(Features_test_i)

    #print section
    #print("The Score svr: %s" %Best_trained_model.score(Features_test, Signal_test))
    #print("Best Hyperparameters: %s" %svr.best_params_)
    timeend = time.time()
    #print("SVR took %s seconds" %(timeend-timestart))
    return {"score" : Best_trained_model.score(Features_test, Signal_test),
            "best_params" : svr.best_params_,
            "prediction" : predicted,
            "ComputationTime" : (timeend-timestart),
            "Best_trained_model": Best_trained_model,
            "feature_importance": "Not available for that model"}

def svr_bayesian_predictor(Features_train, Signal_train, Features_test, Signal_test, HyperparameterGrid, CV, Max_evals, Recursive=False):
    #print("Cell Bayesian Optimization SVR start---------------------------------------------------------")
    timestart = time.time()

    #HyperparameterGrid = {"C": hp.loguniform("C", -6, 23.025), "gamma":hp.loguniform("gamma", -6,23.025), "epsilon":hp.loguniform("epsilon", -6, 23.025)} #if using loguniform, e.g. you want the parameter range 0.1 to 1000 type in (log(0.1), log(1000))

    Signal_train = Signal_train.values.ravel()
    Features_train = Features_train.values

    def hyperopt_cv(params):
        t_start = time.time()
        Estimator = SVR(**params, cache_size=1500)    #give the specific parameter sample per run from fmin
        CV_score = cross_val_score(estimator=Estimator, X=Features_train, y=Signal_train, cv=CV, scoring="r2").mean()  # create a crossvalidation score which shall be optimized
        t_end = time.time()
        print("Params per iteration: %s \ with the cross-validation score %.3f, took %.2fseconds" % (params, CV_score, (t_end-t_start)))
        return CV_score

    def f(params):
        acc = hyperopt_cv(params)
        return {"loss": -acc, "status": STATUS_OK} #fmin always minimizes the loss function, we want acc to maximize-> (-acc)

    trials = Trials() #this is for tracking the bayesian optimization
    BestParams = fmin(f, HyperparameterGrid, algo=tpe.suggest, max_evals=Max_evals, trials=trials) #do the bayesian optimization
    Best_trained_model = SVR(**BestParams).fit(Features_train, Signal_train)    #set the best hyperparameter to the SVR machine
    if Recursive == False:
        predicted = Best_trained_model.predict(Features_test)
    elif Recursive == True:
        Features_test_i = recursive(Features_test, Best_trained_model)
        predicted = Best_trained_model.predict(Features_test_i)

    #print section
    #print("Bayesian Optimization Parameters")
    #print("Everything about the search: %s" %trials.trials)
    #print("List of returns of \"Objective\": %s" %trials.results)
    #print("List of losses per ok trial: %s" %trials.losses())
    #print("List of statuses: %s" %trials.statuses())
    #print("BlackBox Parameter")
    #print("The Score svr: %s" %Best_trained_model.score(Features_test, Signal_test))
    #print("Best Hyperparameters: %s" %BestParams)
    timeend = time.time()
    #print("SVR took %s seconds" %(timeend-timestart))
    return {"score" : Best_trained_model.score(Features_test, Signal_test),
            "best_params" : BestParams,
            "prediction" : predicted,
            "ComputationTime" : (timeend-timestart),
            "Best_trained_model": Best_trained_model,
            "feature_importance": "Not available for that model"}

def rf_predictor(Features_train, Signal_train, Features_test, Signal_test, HyperparameterGrid=NotImplemented, CV=NotImplemented, Max_evals=NotImplemented, Recursive=False):
    #print("Cell RandomForest start---------------------------------------------------------")
    timestart = time.time()

    Signal_test = Signal_test.values.ravel()
    #Features_test = Features_test.values.ravel() #this one not in order to have recursive still working fine
    Signal_train = Signal_train.values.ravel()
    Features_train = Features_train.values

    #using RandomForest
    rf = RandomForestRegressor() #here you could state a max_depth for rf
    Best_trained_model = rf.fit(Features_train, Signal_train)
    if not Features_test.empty: #check whether the test data is not empty #todo:finish(seems to work now but still do for the other models) (maybe better as a class, think an plan)(didnt work because there is no Signal_test for scoring or doing the score; check whether score is necessary anyways, because it is scored later on)(Think of objectoriented programming, there you could apply the function "score" inside the class and only call it if necesaarry
        if Recursive == False:
            predicted = Best_trained_model.predict(Features_test)
        elif Recursive == True:
            Features_test_i = recursive(Features_test, Best_trained_model)
            predicted = Best_trained_model.predict(Features_test_i)
        score = Best_trained_model.score(Features_test, Signal_test)
    else:
        predicted = []
        score = "empty"

    #print section
    print("The Score rf: %s" %score)
    #print("Feature Importance RF: %s" %Best_trained_model.feature_importances_)
    timeend = time.time()
    print("RF took %s seconds" %(timeend-timestart))
    return {"score": score,
            "feature_importance": Best_trained_model.feature_importances_,
            "prediction": predicted,
            "ComputationTime" : (timeend-timestart),
            "Best_trained_model": Best_trained_model,
            "best_params": "Not available for RF"}

def gradientboost_gridsearch(Features_train, Signal_train, Features_test, Signal_test, HyperparameterGrid, CV, Max_evals=NotImplemented, Recursive=False):
    #print("Cell GradientBoost start---------------------------------------------------------")
    timestart = time.time()

    #HyperparameterGrid = [{"n_estimators" : [10,100,1000,10000,100000], "max_depth" : [0.1,1,10,100,1000], "learning_rate" : [0.01,0.1,0.5,1], "loss" : ["ls", "lad", "huber", "quantile"]}]

    #using gradient boosting with gridsearch
    gb = GridSearchCV(GradientBoostingRegressor(), HyperparameterGrid, cv=CV, scoring="r2")
    gb = gb.fit(Features_train, Signal_train)

    # A single gb with the paramaters found in Gridsearch is implemented in order to be able to use the .feature_importances_ attribute and see the influence of the features
    bestgb = GradientBoostingRegressor()
    bestgb = bestgb.set_params(**gb.best_params_)
    Best_trained_model = bestgb.fit(Features_train, Signal_train)
    if Recursive == False:
        predicted = Best_trained_model.predict(Features_test)
    elif Recursive == True:
        Features_test_i = recursive(Features_test, Best_trained_model)
        predicted = Best_trained_model.predict(Features_test_i)

    #print section
    #print("The Score gb: %s" %Best_trained_model.score(Features_test, Signal_test))
    #print("Feature Importance gb: %s" %Best_trained_model.feature_importances_)
    #print("best_params: %s" %gb.best_params_)
    timeend = time.time()
    #print("gb took %s seconds" %(timeend-timestart))
    return {"score": Best_trained_model.score(Features_test, Signal_test),
            "best_params": gb.best_params_,
            "feature_importance": Best_trained_model.feature_importances_,
            "prediction": predicted,
            "ComputationTime" : (timeend-timestart),
            "Best_trained_model": Best_trained_model}

def gradientboost_bayesian(Features_train, Signal_train, Features_test, Signal_test, HyperparameterGrid, CV, Max_evals, Recursive=False):
    #print("Cell Bayesian Optimization GB start---------------------------------------------------------")
    timestart = time.time()

    #HyperparameterGrid = {"n_estimators": scope.int(hp.qloguniform("n_estimators", log(1), log(1e3), 1)),
    #                      "max_depth": scope.int(hp.qloguniform("max_depth", log(1), log(100), 1)),
    #                      "learning_rate": hp.loguniform("learning_rate", log(1e-2), log(1)), "loss": hp.choice("loss",
    #                                                                                                            ["ls",
    #                                                                                                             "lad",
    #                                                                                                             "huber",
    #                                                                                                             "quantile"])}  # if anything except numbers is changed, please change the respective code lines for converting notation style in the gradienboost_bayesian function

    Signal_test = Signal_test.values.ravel()
    #Features_test = Features_test.values.ravel() #this one not in order to have recursive still working fine
    Signal_train = Signal_train.values.ravel()
    Features_train = Features_train.values

    def hyperopt_cv(params):
        t_start = time.time()
        Estimator = GradientBoostingRegressor(**params)    #give the specific parameter sample per run from fmin
        CV_score = cross_val_score(estimator=Estimator, X=Features_train, y=Signal_train, cv=CV, scoring="r2").mean()  # create a crossvalidation score which shall be optimized
        t_end = time.time()
        print("Params per iteration: %s \ with the cross-validation score %.3f, took %.2fseconds" % (params, CV_score, (t_end-t_start)))
        return CV_score

    def f(params):
        acc = hyperopt_cv(params)
        return {"loss": -acc, "status": STATUS_OK} #fmin always minimizes the loss function, we want acc to maximize-> (-acc)

    trials = Trials() #this is for tracking the bayesian optimization
    BestParams = fmin(f, HyperparameterGrid, algo=tpe.suggest, max_evals=Max_evals, trials=trials) #do the bayesian optimization

    #converting notation style
    max_depth = int(BestParams["max_depth"])
    n_estimators = int(BestParams["n_estimators"])
    learning_rate = BestParams["learning_rate"]
    loss = ["ls", "lad", "huber", "quantile"][BestParams["loss"]]
    BestParams = {'learning_rate': learning_rate, 'loss': loss, 'max_depth': max_depth, 'n_estimators': n_estimators}

    Best_trained_model = GradientBoostingRegressor(**BestParams).fit(Features_train, Signal_train)    #set the best hyperparameter to the SVR machine
    if Recursive == False:
        predicted = Best_trained_model.predict(Features_test)
    elif Recursive == True:
        Features_test_i = recursive(Features_test, Best_trained_model)
        predicted = Best_trained_model.predict(Features_test_i)

    #print section
    #print("Bayesian Optimization Parameters")
    #print("Everything about the search: %s" %trials.trials)
    #print("List of returns of \"Objective\": %s" %trials.results)
    #print("List of losses per ok trial: %s" %trials.losses())
    #print("List of statuses: %s" %trials.statuses())
    #print("BlackBox Parameter")
    #print("The Score GB: %s" %Best_trained_model.score(Features_test, Signal_test))
    #print("Best Hyperparameters: %s" %BestParams)
    timeend = time.time()
    #print("GB took %s seconds" %(timeend-timestart))
    return {"score" : Best_trained_model.score(Features_test, Signal_test),
            "feature_importance": Best_trained_model.feature_importances_,
            "best_params" : BestParams,
            "prediction" : predicted,
            "ComputationTime" : (timeend-timestart),
            "Best_trained_model": Best_trained_model}

def lasso_grid_search_predictor(Features_train, Signal_train, Features_test, Signal_test, HyperparameterGrid, CV, Max_evals=NotImplemented, Recursive=False):
    #print("Cell Lasso start----------------------------------------------------------------")
    timestart = time.time()

    #HyperparameterGrid=[{'alpha':[100000, 10, 1, 0.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]}]

    #gridsearch Lasso
    lasso = GridSearchCV(linear_model.Lasso(max_iter=1000000), HyperparameterGrid, cv=CV)
    lasso = lasso.fit(Features_train, Signal_train)

    #A single Lasso with the paramaters found in Gridsearch is implemented in order to be able to use the .coef_ attribute and see the influence of the features
    bestlasso = linear_model.Lasso(max_iter=1000000)
    bestlasso = bestlasso.set_params(**lasso.best_params_)
    Best_trained_model = bestlasso.fit(Features_train, Signal_train)
    if Recursive == False:
        predicted = Best_trained_model.predict(Features_test)
    elif Recursive == True:
        Features_test_i = recursive(Features_test, Best_trained_model)
        predicted = Best_trained_model.predict(Features_test_i)

    #print section
    timeend = time.time()
    #print("The Score Lasso: %s" % Best_trained_model.score(Features_test, Signal_test))
    #print("Best Hyperparameters: %s" %lasso.best_params_)
    #print("Lasso coef: %s" % Best_trained_model.coef_)
    #print("Lasso took %s seconds" %(timeend-timestart))

    return {"score": Best_trained_model.score(Features_test, Signal_test),
            "best_params": lasso.best_params_,
            "feature_importance": Best_trained_model.coef_,
            "prediction": predicted,
            "ComputationTime" : (timeend-timestart),
            "Best_trained_model": Best_trained_model}

def lasso_bayesian(Features_train, Signal_train, Features_test, Signal_test, HyperparameterGrid, CV, Max_evals, Recursive=False):
    #print("Cell Bayesian Optimization Lasso start---------------------------------------------------------")
    timestart = time.time()

    #HyperparameterGrid = {"alpha": hp.loguniform("alpha", log(1e-10), log(1e6))}
    Signal_test = Signal_test.values.ravel()
    #Features_test = Features_test.values.ravel() #this one not in order to have recursive still working fine
    Signal_train = Signal_train.values.ravel()
    Features_train = Features_train.values

    def hyperopt_cv(params):
        t_start = time.time()
        Estimator = linear_model.Lasso(**params, max_iter=1000000)    #give the specific parameter sample per run from fmin
        CV_score = cross_val_score(estimator=Estimator, X=Features_train, y=Signal_train, cv=CV, scoring="r2").mean()  # create a crossvalidation score which shall be optimized
        t_end = time.time()
        print("Params per iteration: %s \ with the cross-validation score %.3f, took %.2fseconds" % (params, CV_score, (t_end-t_start)))
        return CV_score

    def f(params):
        acc = hyperopt_cv(params)
        return {"loss": -acc, "status": STATUS_OK} #fmin always minimizes the loss function, we want acc to maximize-> (-acc)

    trials = Trials() #this is for tracking the bayesian optimization
    BestParams = fmin(f, HyperparameterGrid, algo=tpe.suggest, max_evals=Max_evals, trials=trials) #do the bayesian optimization
    Best_trained_model = linear_model.Lasso(**BestParams, max_iter=1000000).fit(Features_train, Signal_train)    #set the best hyperparameter to the SVR machine
    if Recursive == False:
        predicted = Best_trained_model.predict(Features_test)
    elif Recursive == True:
        Features_test_i = recursive(Features_test, Best_trained_model)
        predicted = Best_trained_model.predict(Features_test_i)

    #print section
    #print("Bayesian Optimization Parameters")
    #print("Everything about the search: %s" %trials.trials)
    #print("List of returns of \"Objective\": %s" %trials.results)
    #print("List of losses per ok trial: %s" %trials.losses())
    #print("List of statuses: %s" %trials.statuses())
    #print("BlackBox Parameter")
    #print("The Score Lasso: %s" %Best_trained_model.score(Features_test, Signal_test))
    #print("Best Hyperparameters: %s" %BestParams)
    timeend = time.time()
    #print("Lasso took %s seconds" %(timeend-timestart))
    return {"score" : Best_trained_model.score(Features_test, Signal_test),
            "feature_importance": Best_trained_model.coef_,
            "best_params" : BestParams,
            "prediction" : predicted,
            "ComputationTime" : (timeend-timestart),
            "Best_trained_model": Best_trained_model}

def ann_grid_search_predictor(Features_train, Signal_train, Features_test, Signal_test, HyperparameterGrid, CV, Max_evals=NotImplemented, Recursive=False):
    #print("Cell GridSearchANN start---------------------------------------------------------")
    timestart = time.time()

    #HyperparameterGrid= [{'hidden_layer_sizes':[[1],[10],[100],[1000],[1, 1],[10, 10], [100, 100],[1,10],[1,100],[10,100],[100,10],[100,1],[10,1],[1, 1, 1],[10, 10, 10],[100,100,100]]}]

    #gridsearch with MLP
    ann = GridSearchCV(MLPRegressor(max_iter = 1000000), HyperparameterGrid, cv=CV)
    Best_trained_model = ann.fit(Features_train, Signal_train)
    if Recursive == False:
        predicted = Best_trained_model.predict(Features_test)
    elif Recursive == True:
        Features_test_i = recursive(Features_test, Best_trained_model)
        predicted = Best_trained_model.predict(Features_test_i)

    timeend = time.time()
    #print section
    #print("The Score ann: %s" %Best_trained_model.score(Features_test, Signal_test))
    #print("Best Hyperparameters: %s" %ann.best_params_)
    #print("ANN took %s seconds" %(timeend-timestart))

    return {"score" : Best_trained_model.score(Features_test, Signal_test),
            "best_params" : ann.best_params_,
            "prediction" : predicted,
            "ComputationTime" : (timeend-timestart),
            "Best_trained_model": Best_trained_model,
            "feature_importance": "Not available for that model"}

def ann_bayesian_predictor(Features_train, Signal_train, Features_test, Signal_test, HyperparameterGrid, CV, Max_evals, Recursive=False):
    #print("Cell Bayesian Optimization ANN start---------------------------------------------------------")
    timestart = time.time()

    #HyperparameterGrid= hp.choice("number_of_layers",
    #                    [
    #                    {"1layer": scope.int(hp.qloguniform("1.1", log(1), log(1000), 1))},
    #                    {"2layer": [scope.int(hp.qloguniform("1.2", log(1), log(1000), 1)), scope.int(hp.qloguniform("2.2", log(1), log(1000), 1))]},
    #                    {"3layer": [scope.int(hp.qloguniform("1.3", log(1), log(1000), 1)), scope.int(hp.qloguniform("2.3", log(1), log(1000), 1)), scope.int(hp.qloguniform("3.3", log(1), log(1000), 1))]}
    #                    ])

    Signal_test = Signal_test.values.ravel()
    #Features_test = Features_test.values.ravel() #this one not in order to have recursive still working fine
    Signal_train = Signal_train.values.ravel()
    Features_train = Features_train.values

    def hyperopt_cv(params):
        t_start = time.time()
        try: #set params so that it fits the estimators attribute style
            params = {"hidden_layer_sizes": params["1layer"]}
        except:
            try:
                params = {"hidden_layer_sizes": params["2layer"]}
            except:
                try:
                    params = {"hidden_layer_sizes": params["3layer"]}
                except:
                    sys.exit("Your bayesian hyperparametergrid does not fit the requirements, check the example and/or change the hyperparametergrid or the postprocessing in def hyperopt_cv")
        Estimator = MLPRegressor(**params, max_iter=1000000)    #give the specific parameter sample per run from fmin
        CV_score = cross_val_score(estimator=Estimator, X=Features_train, y=Signal_train, cv=CV, scoring="r2").mean()  # create a crossvalidation score which shall be optimized
        t_end = time.time()
        print("Params per iteration: %s \ with the cross-validation score %.3f, took %.2fseconds" % (params, CV_score, (t_end-t_start)))
        return CV_score

    def f(params):
        acc = hyperopt_cv(params)
        return {"loss": -acc, "status": STATUS_OK} #fmin always minimizes the loss function, we want acc to maximize-> (-acc)

    trials = Trials()
    BestParams = fmin(f, HyperparameterGrid, algo=tpe.suggest, max_evals=Max_evals, trials=trials)
    try: #set params so that it fits the estimators attribute style
        Z = [int(BestParams["1.1"])]
    except:
        try:
            Z = [int(BestParams["1.2"]), int(BestParams["2.2"])]
        except:
            try:
                Z = [int(BestParams["1.3"]), int(BestParams["2.3"]), int(BestParams["2.3"])]
            except:
                sys.exit("Your bayesian hyperparametergrid does not fit the requirements, check the example and/or change the hyperparametergrid or the postprocessing for the bestparams in ann_bayesian_predictor")
    BestParams = {"hidden_layer_sizes": Z} #set params so that it fits the estimators attribute style
    Ann_best = MLPRegressor(**BestParams)    #set the best hyperparameter to the SVR machine
    Best_trained_model = Ann_best.fit(Features_train, Signal_train)
    if Recursive == False:
        predicted = Best_trained_model.predict(Features_test)
    elif Recursive == True:
        Features_test_i = recursive(Features_test, Best_trained_model)
        predicted = Best_trained_model.predict(Features_test_i)

    #print section
    #print("Bayesian Optimization Parameters")
    #print("Everything about the search: %s" %trials.trials)
    #print("List of returns of \"Objective\": %s" %trials.results)
    #print("List of losses per ok trial: %s" %trials.losses())
    #print("List of statuses: %s" %trials.statuses())
    #print("BlackBox Parameter")
    #print("The Score ann: %s" %Ann_best.score(Features_test, Signal_test))
    #print("Best Hyperparameters: %s" %BestParams)
    timeend = time.time()
    #print("ANN took %s seconds" %(timeend-timestart))
    return {"score" : Ann_best.score(Features_test, Signal_test),
            "best_params" : BestParams,
            "prediction" : predicted,
            "ComputationTime" : (timeend-timestart),
            "Best_trained_model": Best_trained_model,
            "feature_importance": "Not available for that model"}

#Individual Models------------------------------------------------------------------------------------------------------
#Splitter functions
def week_weekend_splitter(Dataseries):
    # Datetimeindex format is necessary for individual model methods
    # select all weekday and weekend days from the specific dataseries
    weekday = Dataseries[Dataseries.index.dayofweek <= 4]  # here you can change the week / weekend definition
    weekend = Dataseries[Dataseries.index.dayofweek >= 5]
    Dic = {"weekday": weekday, "weekend": weekend}
    return Dic

def hourly_splitter(Dataseries):
    Dic = dict()
    #select all values from each respective hour and add the to a dictionary
    for hour in range(0,24):
        hourly = Dataseries[Dataseries.index.hour == hour]
        Dic.update({hour: hourly})
    return Dic

class byfeature_splitter():
    'Class for the "byFeature" splitter as with a class the two additional attributes "indivFeature" and "Threshold" can be propagated throughout all following computations'
    def __init__(self, Threshold, Feature, Features_Test, Features_Train="optional"):
        self.Threshold = Threshold
        self.Feature = Feature
        self.Features_Train = Features_Train
        self.Features_Test = Features_Test
        if type(self.Features_Train) != str:
            self.idx_train_above = Features_Train.index[Features_Train[self.Feature] >= self.Threshold]
            self.idx_train_below = Features_Train.index[Features_Train[self.Feature] < self.Threshold]
        self.idx_test_above = Features_Test.index[Features_Test[self.Feature] >= self.Threshold]
        self.idx_test_below = Features_Test.index[Features_Test[self.Feature] < self.Threshold]

    def splitter(self, Dataseries):
        # Datetimeindex format is necessary for individual model methods
        # select all above the stated threshold
        if type(self.Features_Train) != str:
            if Dataseries.index.equals(self.Features_Train.index): #check whether Dataseries is within train or test period
                above = Dataseries.loc[self.idx_train_above]
                below = Dataseries.loc[self.idx_train_below]
        if Dataseries.index.equals(self.Features_Test.index):
            above = Dataseries.loc[self.idx_test_above]
            below = Dataseries.loc[self.idx_test_below]

        Dic = {"above": above, "below": below}
        return Dic

#Individual Models executive functions
class indiv_splitter():
    'Splits the dataframe with the respective "Split_function" in the dataframes needed for training the individual models. The dataframes are safed as needed from the "indiv_model" and "indiv_model_onlypredict" classes'
    def __init__(self, Split_function):
        self.Split_function = Split_function

    def split_train_test(self, Features_train, Signal_train, Features_test, Signal_test):
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

class indiv_model():
    'Trains the indivdual models and does a prediction'
    def __init__(self, indiv_splitter_instance, Estimator, Features_train, Signal_train, Features_test, Signal_test, HyperparameterGrid=None, CV=None, Max_evals=None, Recursive=False):
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
        Dic = self.indiv_splitter_instance.split_train_test(self.Features_train, self.Signal_train, self.Features_test, self.Signal_test)

        best_params = dict()
        best_model = dict()

        Y = pd.DataFrame(index=self.Signal_test.index)
        i = 1
        for key in Dic:
            if Dic[key][0].empty:
                Answer = input(
                    "Attention your train period does not contain data to train all individual models. Proceed anyways?")
                if Answer == "yes" or Answer == "Yes" or Answer == "y" or Answer == "Y":
                    print("Start computing")
                else:
                    sys.exit("Code stopped by user or invalid user input. Valid is Yes, yes, y and Y.")
            #Todo: if not Dic[key][3].empty:  # to avoid a crash if not all individual models are called in the test data range #TODO: makes the model not train for times where no test period is available, if telling to train it crashes because i
            _dic = self.Estimator(Features_train=Dic[key][0], Signal_train=Dic[key][1], Features_test=Dic[key][2],
                             Signal_test=Dic[key][3], HyperparameterGrid=self.HyperparameterGrid, CV=self.CV, Max_evals=self.Max_evals,
                             Recursive=False)  # train and predict for the given data #recursive has to be turned of (doesnt work with individual model), it is done later in this function for individual models
            Y_i = _dic["prediction"]  # pull the prediction from the dictionary
            Index = Dic[key][3]
            Y_i = pd.DataFrame(index=Index.index, data=Y_i)  # reset the index to datetime convention
            Y_i = Y_i.rename(columns={0: i})  # rename column per loop to have each period in a single column
            Y = pd.concat([Y, Y_i], axis=1)  # add them all together
            i += 1
            try:  # try to add the best hyperparameters per time intervall, try is necessary since not all estimators pass "best_params"
                best_params[key] = _dic["best_params"]
            except:
                pass
            best_model[key] = _dic["Best_trained_model"]  # add best models to a list

        if self.Recursive == False:
            predicted = Y.sum(
                axis=1)  # add all columns together, since each timestamp has only 1 column with a value this is the same as rearranging all the results back to a chronological timeline
        if self.Recursive == True:
            Features_test_i = self.Features_test.copy(deep=True)
            Features_test_i.index = range(len(Features_test_i))  # set an trackable index 0,1,2,3,etc.
            Features_test_ii = self.Features_test.copy(deep=True)
            Features_test_ii["TrackIndex"] = range(len(
                self.Features_test))  # add an trackable index to the original one #just for tracking the index of Features_test_i

            #split Features_test_ii into individual model sets
            Dic = self.indiv_splitter_instance.split_test(Features_test_ii)

            for i in Features_test_i.index:
                vector_i = Features_test_i.iloc[[i]]  # get the features of the timestep i

                #if i is in one of the dic[key] data sets, use this key!
                for key in Dic: #loop through all dictionary entries
                    if not Dic[key].empty:# to avoid a crash if not all individual models are called in the test data range
                        if i in Dic[key].set_index("TrackIndex").index: #checks whether the line i is in the data for the data of the respective key
                            OwnLag = best_model[key].predict(vector_i)  # do a one one timestep prediction with the model of the respective key

                Booleans = Features_test_i.columns.str.contains("_lag_")  # create a Boolean list for with all columns, true for lagged signals, false for other(important: for lagged features it is only "_lag"
                Lagged_column_list = np.array(list(Features_test_i))[Booleans]
                for columnname in Lagged_column_list:  # go through each column containing _lag_ in its name
                    lag = columnname.split("_")[-1]  # get the lag from the name of the column (lagged signals have the ending, e.g. for lag 1:  "_lag_1"
                    line = int(lag) + i  # define the line where the specific prediction should be safed
                    if line < len(Features_test_i): #save produced ownlag in features_test_i
                        Features_test_i = Features_test_i.set_value(value=OwnLag, index=line, col=Features_test_i.columns.str.contains("_lag_%s" % lag))  # set the predicted signal as input for future predictions

            Features_test_i = Features_test_i.set_index(self.Signal_test.index)#pd.DataFrame(index=Signal_test.index, data=Features_test_i)  # reset the index to datetime convention
            #split the recursive features_test_i up
            Dic = self.indiv_splitter_instance.split_test(Features_test_i)

            #do an individual model prediction for the "recursive" feature set: Features_test_i
            Y = pd.DataFrame(index=self.Signal_test.index)
            i = 1
            for key in Dic:
                if not Dic[key].empty:  # to avoid a crash if not all individual models are called in the test data range
                    Y_i = best_model[key].predict(Dic[key])
                    Index = Dic[key]
                    Y_i = pd.DataFrame(index=Index.index, data=Y_i)  # reset the index to datetime convention
                    Y_i = Y_i.rename(columns={0: i})  # rename column per loop to have each period in a single column
                    Y = pd.concat([Y, Y_i], axis=1)  # add them all together
                    i += 1
            predicted = Y.sum(axis=1)  # add all columns together, since each timestamp has only 1 column with a value this is the same as rearranging all the results back to a chronological timeline

        timeend=time.time()
        return {"prediction": predicted,
                "best_params": best_params,
                "ComputationTime" : (timeend-timestart),
                "Best_trained_model": best_model,
                "feature_importance": "Not available for individual model"
                }

class indiv_model_onlypredict():
    'Loads a beforehand safed (individual) model and does a prediction'
    def __init__(self, indiv_splitter_instance, Features_test, ResultsFolderSubTest, NameOfPredictor, Recursive):
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
                if not Dic[key].empty:  # to avoid a crash if not all individual models are called in the test data range
                    Predictor = joblib.load("%s\\BestModels\\%s_%s.save" % (self.ResultsFolderSubTest, key, self.NameOfPredictor))
                    Y_i = Predictor.predict(Dic[key])  # predict
                    Index = Dic[key]
                    Y_i = pd.DataFrame(index=Index.index, data=Y_i)  # reset the index to datetime convention
                    Y_i = Y_i.rename(columns={0: i})  # rename column per loop to have each period in a single column
                    Y = pd.concat([Y, Y_i], axis=1)  # add them all together
                    i += 1
            predicted = Y.sum(axis=1)  # add all columns together, since each timestamp has only 1 column with a value this is the same as rearranging all the results back to a chronological timeline
        if self.Recursive == True:
            Features_test_i = self.Features_test.copy(deep=True)
            Features_test_i.index = range(len(Features_test_i))  # set an trackable index 0,1,2,3,etc.
            Features_test_ii = self.Features_test.copy(deep=True)
            Features_test_ii["TrackIndex"] = range(len(self.Features_test))  # add an trackable index to the original one #just for tracking the index of Features_test_i

            # split Features_test_ii into individual model sets
            Dic = self.indiv_splitter_instance.split_test(Features_test_ii)

            for i in Features_test_i.index:
                vector_i = Features_test_i.iloc[[i]]  # get the features of the timestep i

                # if i is in one of the dic[key] data sets, use this key!
                for key in Dic:  # loop through all dictionary entries
                    if not Dic[key].empty:  # to avoid a crash if not all individual models are called in the test data range
                        if i in Dic[key].set_index("TrackIndex").index:  # checks whether the line i is in the data for the data of the respective key
                            Predictor = joblib.load("%s\\BestModels\\%s_%s.save" % (self.ResultsFolderSubTest, key, self.NameOfPredictor))  # load the respective model
                            OwnLag = Predictor.predict(vector_i)  # do a one one timestep prediction with the model of the respective key

                Booleans = Features_test_i.columns.str.contains("_lag_")  # create a Boolean list for with all columns, true for lagged signals, false for other(important: for lagged features it is only "_lag"
                Lagged_column_list = np.array(list(Features_test_i))[Booleans]
                for columnname in Lagged_column_list:  # go through each column containing _lag_ in its name
                    lag = columnname.split("_")[-1]  # get the lag from the name of the column (lagged signals have the ending, e.g. for lag 1:  "_lag_1"
                    line = int(lag) + i  # define the line where the specific prediction should be safed
                    if line < len(Features_test_i):  # save produced ownlag in features_test_i
                        Features_test_i = Features_test_i.set_value(value=OwnLag, index=line,
                                                                    col=Features_test_i.columns.str.contains(
                                                                        "_lag_%s" % lag))  # set the predicted signal as input for future predictions

            Features_test_i = Features_test_i.set_index(Datetimetracker.index)  # pd.DataFrame(index=Signal_test.index, data=Features_test_i)  # reset the index to datetime convention

            # split the recursive features_test_i up
            Dic = self.indiv_splitter_instance.split_test(Features_test_i)

            # do an individual model prediction for the "recursive" feature set: Features_test_i
            Y = pd.DataFrame(index=Datetimetracker.index)
            i = 1
            for key in Dic:
                if not Dic[key].empty:  # to avoid a crash if not all individual models are called in the test data range
                    Predictor = joblib.load("%s\\BestModels\\%s_%s.save" % (self.ResultsFolderSubTest, key, self.NameOfPredictor))  # load the respective model
                    Y_i = Predictor.predict(Dic[key])
                    Index = Dic[key]
                    Y_i = pd.DataFrame(index=Index.index, data=Y_i)  # reset the index to datetime convention
                    Y_i = Y_i.rename(columns={0: i})  # rename column per loop to have each period in a single column
                    Y = pd.concat([Y, Y_i], axis=1)  # add them all together
                    i += 1
            predicted = Y.sum(axis=1)  # add all columns together, since each timestamp has only 1 column with a value this is the same as rearranging all the results back to a chronological timeline
        return predicted
#-----------------------------------------------------------------------------------------------------------------------



from math import log
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV

from BlackBoxes import *
from Functions.ErrorMetrics import *

import ModelTuning as MT
import SharedVariablesFunctions as SVF
from ModelTuningRuntimeResults import ModelTuningRuntimeResults as MTRR
from core.model_tuning.config.model_tuning_config import ModelTuningSetup as MTS


def embedded__recursive_feature_selection(
    MT_Setup_Object,
    _X_train,
    _Y_train,
    _X_test,
    _Y_test,
    Estimator,
    N_features_to_select,
    CV,
    Documentation=False,
):
    # Special feature selection method for the feature selection within the final bayesian optimization (def Bayes())
    def index_column_keeper(X_Data, Y_Data, support, X_Data_transformed):
        columns = X_Data.columns
        rows = X_Data.index
        labels = [
            columns[x] for x in support if x >= 0
        ]  # get the columns which shall be kept by the transformer(the selected features)
        X = pd.DataFrame(
            X_Data_transformed, columns=labels, index=rows
        )  # creates a dataframe reassigning the names of the features as column header and the index as index
        Y = pd.DataFrame(
            Y_Data, columns=[MT_Setup_Object.name_of_target]
        )  # create dataframe of y
        return X, Y

    if N_features_to_select == "automatic":
        selector = RFECV(estimator=Estimator, step=1, cv=CV)
        selector = selector.fit(_X_train, _Y_train)
        print("Ranks of all Features %s" % selector.ranking_)
        Features_transformed = selector.transform(_X_train)
        Features_transformed_test = selector.transform(_X_test)
        Features_transformed, _Y_train = index_column_keeper(
            _X_train, _Y_train, selector.get_support(indices=True), Features_transformed
        )
        Features_transformed_test, _Y_test = index_column_keeper(
            _X_test,
            _Y_test,
            selector.get_support(indices=True),
            Features_transformed_test,
        )
    else:
        selector = RFE(
            estimator=Estimator, n_features_to_select=N_features_to_select, step=1
        )
        selector = selector.fit(_X_train, _Y_train)
        print("Ranks of all Features %s" % selector.ranking_)
        Features_transformed = selector.transform(_X_train)
        Features_transformed_test = selector.transform(_X_test)
        Features_transformed, _Y_train = index_column_keeper(
            _X_train, _Y_train, selector.get_support(indices=True), Features_transformed
        )
        Features_transformed_test, _Y_test = index_column_keeper(
            _X_test,
            _Y_test,
            selector.get_support(indices=True),
            Features_transformed_test,
        )

    if Documentation == False:
        return Features_transformed, _Y_train, Features_transformed_test, _Y_test
    if Documentation == True:

        def merge_signal_and_features_embedded(
            X_Data, Y_Data, support, X_Data_transformed
        ):  # Todo: could be pulled directly from SharedVariables (check for how to get the right "NameOfSignal"
            columns = X_Data.columns
            rows = X_Data.index
            labels = [
                columns[x] for x in support if x >= 0
            ]  # get the columns which shall be kept by the transformer(the selected features)
            Features = pd.DataFrame(
                X_Data_transformed, columns=labels, index=rows
            )  # creates a dataframe reassigning the names of the features as column header and the index as index
            Signal = pd.DataFrame(
                Y_Data, columns=[MT_Setup_Object.name_of_target]
            )  # create dataframe of y
            Data = pd.concat([Signal, Features], axis=1)
            return Data

        _Data_Train = merge_signal_and_features_embedded(
            _X_train, _Y_train, selector.get_support(indices=True), Features_transformed
        )  # merge signal and features
        _Data_Test = merge_signal_and_features_embedded(
            _X_test,
            _Y_test,
            selector.get_support(indices=True),
            Features_transformed_test,
        )  # merge signal and features
        BestData = pd.concat(
            [_Data_Train, _Data_Test], axis=0
        )  # merge test and fit period back together
        return (
            Features_transformed,
            _Y_train,
            Features_transformed_test,
            _Y_test,
            BestData,
        )


def Bayes(
    MT_Setup_Object_AFB,
    MT_RR_object_AFB,
    _X_train,
    _Y_train,
    _X_test,
    _Y_test,
    Indexer,
    Data,
):
    # Here the final bayesian optimization is done
    Model = MT_Setup_Object_AFB.Model_Bayes
    Totaltimestart = time.time()
    if Model == "Baye":  # set the bayesian parameter space
        params = {
            "IndivModel": hp.choice(
                "IndivModel",
                [
                    {"IndivModel_baye": "No"},
                    {"IndivModel_baye": "hourly"},
                    {"IndivModel_baye": "week_weekend"},
                ],
            ),
            "n_F": hp.qloguniform("n_F", log(1), log(len(list(_X_test))), 1),
            "Model": hp.choice(
                "Model",
                [
                    {"Model": "SVR"},
                    {"Model": "ANN"},
                    {"Model": "GB"},
                    {"Model": "RF"},
                    {"Model": "Lasso"},
                ],
            ),
        }
    else:
        params = {
            "IndivModel": hp.choice(
                "IndivModel",
                [
                    {"IndivModel_baye": "No"},
                    {"IndivModel_baye": "hourly"},
                    {"IndivModel_baye": "week_weekend"},
                ],
            ),
            "n_F": hp.qloguniform("n_F", log(1), log(len(list(_X_test))), 1),
        }

    """
    #Todo: just for checking; delete afterwards
    import hyperopt.pyll
    print (hyperopt.pyll.stochastic.sample(params))
    print (hyperopt.pyll.stochastic.sample(params))
    print (hyperopt.pyll.stochastic.sample(params))
    print (hyperopt.pyll.stochastic.sample(params))
    print (hyperopt.pyll.stochastic.sample(params))
    print (hyperopt.pyll.stochastic.sample(params))
    print (hyperopt.pyll.stochastic.sample(params))
    print (hyperopt.pyll.stochastic.sample(params))
    print (hyperopt.pyll.stochastic.sample(params))
    print (hyperopt.pyll.stochastic.sample(params))
    print (hyperopt.pyll.stochastic.sample(params))
    """

    def hyperopt(params, _X_train, _Y_train, _X_test, _Y_test, Indexer):
        t_start = time.time()
        if (
            Model == "Baye"
        ):  # if model is chosen by bayesian optimization, set Model equal to the one from the params
            _Model = params["Model"]["Model"]
        else:
            _Model = Model

        EstimatorEmbedded = (
            SVF.rf
        )  # rf is a shared variable defined in SharedVariables.py

        (_X_train, _Y_train, _X_test, _Y_test) = embedded__recursive_feature_selection(
            MT_Setup_Object_AFB,
            _X_train,
            _Y_train,
            _X_test,
            _Y_test,
            EstimatorEmbedded,
            params["n_F"],
            MT_Setup_Object_AFB.GlobalCV_MT,
        )  # create the specific fit and test data

        _Model = [_Model]  # converting string to list
        print("The model sent is", _Model)
        Score = train_predict_selected_models(
            MT_Setup_Object_AFB,
            MT_RR_object_AFB,
            _Model,
            _X_train,
            _Y_train,
            _X_test,
            _Y_test,
            Indexer,
            str(params["IndivModel"]["IndivModel_baye"]),
            False,
        )
        t_end = time.time()
        print(
            "Params per iteration: %s \ with the Score score_test %.3f, took %.2fseconds"
            % (params, Score, (t_end - t_start))
        )
        return Score

    def f(params):
        acc = hyperopt(
            params, _X_train, _Y_train, _X_test, _Y_test, Indexer
        )  # gets the score_test of the model
        return {
            "loss": -acc,
            "status": STATUS_OK,
        }  # fmin always minimizes the loss function, we want acc to maximize-> (-acc)

    # do the actual bayesian optimization
    trials = (
        Trials()
    )  # not used at the moment, only for tracking the intrinsic parameters of the bayesian optimization
    BestParams = fmin(
        f,
        params,
        algo=tpe.suggest,
        max_evals=MT_Setup_Object_AFB.MaxEval_Bayes,
        trials=trials,
    )  # Do the optimization to find the best settings(parameters)

    # converting notation style
    if Model == "Baye":
        Best_IndivModel = ["No", "hourly", "week_weekend"][BestParams["IndivModel"]]
        Best_Model = ["SVR", "ANN", "GB", "RF", "Lasso"][BestParams["Model"]]
        Best_n_F = BestParams["n_F"]
        BestParams = {
            "IndivModel": {"IndivModel_baye": Best_IndivModel},
            "Model": {"Model": Best_Model},
            "n_F": Best_n_F,
        }
    else:
        Best_IndivModel = ["No", "hourly", "week_weekend"][BestParams["IndivModel"]]
        Best_n_F = BestParams["n_F"]
        BestParams = {
            "IndivModel": {"IndivModel_baye": Best_IndivModel},
            "n_F": Best_n_F,
        }

    # redo the training and testing with the found "BestParams", also document the results
    if (
        Model == "Baye"
    ):  # if model is chosen by bayesian optimization, set Model equal to the one from the bestparams
        _Model = BestParams["Model"]["Model"]
    else:
        _Model = Model

    EstimatorEmbedded = SVF.rf

    (
        _X_train,
        _Y_train,
        _X_test,
        _Y_test,
        BestData,
    ) = embedded__recursive_feature_selection(
        MT_Setup_Object_AFB,
        _X_train,
        _Y_train,
        _X_test,
        _Y_test,
        EstimatorEmbedded,
        BestParams["n_F"],
        MT_Setup_Object_AFB.GlobalCV_MT,
        True,
    )

    # Todo: Here you could use higher Max_eval for the last final training with best settings(Add specific max eval hyparatuning to the functions)
    _Model = [_Model]  # converting string to list
    Score = train_predict_selected_models(
        MT_Setup_Object_AFB,
        MT_RR_object_AFB,
        _Model,
        _X_train,
        _Y_train,
        _X_test,
        _Y_test,
        Indexer,
        str(BestParams["IndivModel"]["IndivModel_baye"]),
        True,
    )

    # Document the Results and settings of the final bayesian optimization
    Totaltimeend = time.time()
    # save summary of setup and evaluation
    dfSummary = pd.DataFrame(index=[0])
    dfSummary["Chosen Model"] = Model
    dfSummary["Max evaluations"] = MT_Setup_Object_AFB.MaxEval_Bayes
    if Model == "Baye":
        dfSummary["Best Model"] = _Model
    dfSummary["Best individual model type"] = Best_IndivModel
    dfSummary["Best number of features"] = Best_n_F
    dfSummary["Best Features incl. Signal"] = str(list(BestData))
    dfSummary["Best parameter in original shape"] = str(BestParams)
    dfSummary["Computation Time in seconds"] = str((Totaltimeend - Totaltimestart))
    dfSummary = dfSummary.T
    # write summary of setup and evaluation in excel File
    SummaryFile = os.path.join(
        MT_Setup_Object_AFB.ResultsFolderSubTest,
        "Summary_FinalBayes_%s.xlsx" % (MT_Setup_Object_AFB.name_of_model_tuning_experiment),
    )
    writer = pd.ExcelWriter(SummaryFile)
    dfSummary.to_excel(writer, float_format="%.6f")
    writer.save()

    # export BestData to Excel
    BestData = Data[
        list(BestData)
    ]  # make sure BestData contains the whole available period(not only the period used for training and prediction)
    SaveFileName_excel = os.path.join(
        MT_Setup_Object_AFB.ResultsFolderSubTest,
        "BestData_%s.xlsx" % (MT_Setup_Object_AFB.name_of_model_tuning_experiment),
    )
    BestData.to_excel(SaveFileName_excel)

    # save dataframe in an pickle
    BestData.to_pickle(
        os.path.join(
            MT_Setup_Object_AFB.path_to_pickles,
            "ThePickle_from_%s.pickle" % MT_Setup_Object_AFB.name_of_model_tuning_experiment,
        )
    )


def main_FinalBayes(MT_Setup_Object_AFB):
    # The automatic procedure for model tuning and parts of data tuning
    print(
        "Start FinalBayesOpt: %s/%s/%s"
        % (
            MT_Setup_Object_AFB.name_of_raw_data,
            MT_Setup_Object_AFB.name_of_tuning,
            MT_Setup_Object_AFB.name_of_model_tuning_experiment,
        )
    )

    _X_train, _Y_train, _X_test, _Y_test, Indexer, Data = MT.pre_handling(
        MT_Setup_Object_AFB, False
    )

    MT_RR_object_AFB = MTRR()

    # Do the bayesian optimization
    Bayes(
        MT_Setup_Object_AFB,
        MT_RR_object_AFB,
        _X_train=_X_train,
        _Y_train=_Y_train,
        _X_test=_X_test,
        _Y_test=_Y_test,
        Indexer=Indexer,
        Data=Data,
    )

    print(
        "Finish FinalBayesOpt: %s/%s/%s"
        % (
            MT_Setup_Object_AFB.name_of_raw_data,
            MT_Setup_Object_AFB.name_of_tuning,
            MT_Setup_Object_AFB.name_of_model_tuning_experiment,
        )
    )
    print("________________________________________________________________________\n")
    print("________________________________________________________________________\n")
    MT_RR_object_AFB.store_results(MT_Setup_Object_AFB)


if __name__ == "__main__":
    MT_Setup_object_AFB = MTS()
    MT_Setup_object_AFB = SVF.setup_object_initializer(MT_Setup_object_AFB).mts()
    main_FinalBayes(MT_Setup_object_AFB)

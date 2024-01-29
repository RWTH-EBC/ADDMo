from openpyxl import load_workbook
from PredictorDefinitions import *
from Functions.ErrorMetrics import *
from Functions.PlotFcn import *
from core.data_tuning_optimizer.config.data_tuning_config import DataTuningSetup


def documentation_DataTuning(DT_Setup_object: DataTuningSetup, timestart, timeend):
    print("Documentation")
    # dump the name of signal in the resultsfolder, so that you can always be pulled whenever you want to come back to that specific "Final Input Data"
    joblib.dump(
        DT_Setup_object.name_of_target,
        os.path.join(DT_Setup_object.abs_path_to_result_folder, "NameOfSignal.save"),
    )

    # saving the methodology of creating FinalInputData in the ExcelFile "Settings"
    DfMethodology = pd.DataFrame(
        index=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        columns=[
            "GlobalVariables",
            "ImportData",
            "Preprocessing",
            "PeriodSelection",
            "FeatureConstruction",
            "FeatureSelection",
        ],
    )
    # adding information to the dataframe
    DfMethodology.at[1, "GlobalVariables"] = (
        "NameOfData = %s" % DT_Setup_object.name_of_raw_data
    )
    DfMethodology.at[2, "GlobalVariables"] = (
        "NameOfExperiment = %s" % DT_Setup_object.name_of_data_tuning_experiment
    )
    DfMethodology.at[3, "GlobalVariables"] = (
        "NameOfSignal = %s" % DT_Setup_object.name_of_target
    )
    DfMethodology.at[4, "GlobalVariables"] = (
        "Model used for Wrappers = %s" % DT_Setup_object.wrapper_model.__name__
    )
    DfMethodology.at[5, "GlobalVariables"] = (
        "Parameter used for Wrappers = %s" % DT_Setup_object.wrapper_params
    )
    DfMethodology.at[6, "GlobalVariables"] = (
        "MinIncrease for Wrappers = %s" % DT_Setup_object.min_increase_4_wrapper
    )
    DfMethodology.at[7, "GlobalVariables"] = "Pipeline took %s seconds" % (
        timeend - timestart
    )

    DfMethodology.at[1, "Preprocessing"] = (
        "How to deal NaN´s = %s" % DT_Setup_object.nan_filling
    )
    DfMethodology.at[2, "Preprocessing"] = (
        "Initial feature select = %s" % DT_Setup_object.initial_manual_feature_selection
    )
    if DT_Setup_object.initial_manual_feature_selection == True:
        DfMethodology.at[3, "Preprocessing"] = (
            "Features selected = %s" % DT_Setup_object.initial_manual_feature_selection_features
        )
    DfMethodology.at[4, "Preprocessing"] = (
        "StandardScaler = %s" % DT_Setup_object.standard_scaling
    )
    DfMethodology.at[5, "Preprocessing"] = (
        "RobustScaler = %s" % DT_Setup_object.robust_scaling
    )
    DfMethodology.at[6, "Preprocessing"] = "NoScaling = %s" % DT_Setup_object.no_scaling
    DfMethodology.at[7, "Preprocessing"] = "Resample = %s" % DT_Setup_object.resample
    if DT_Setup_object.resample == True:
        DfMethodology.at[8, "Preprocessing"] = (
            "Resolution = %s" % DT_Setup_object.resolution
        )
        DfMethodology.at[9, "Preprocessing"] = (
            "WayOfResampling = %s" % DT_Setup_object.way_of_resampling
        )

    DfMethodology.at[1, "PeriodSelection"] = (
        "ManualSelection = %s" % DT_Setup_object.manual_period_selection
    )
    if DT_Setup_object.manual_period_selection == True:
        DfMethodology.at[2, "PeriodSelection"] = "%s till %s" % (
            DT_Setup_object.start_date,
            DT_Setup_object.end_date,
        )
    DfMethodology.at[3, "PeriodSelection"] = (
        "TimeSeriesPlot = %s" % DT_Setup_object.timeseries_plot
    )

    DfMethodology.at[1, "FeatureConstruction"] = (
        "Cross, auto, cloud correlation plot= %s"
        % DT_Setup_object.correlation_plotting
    )
    if DT_Setup_object.correlation_plotting == True:
        DfMethodology.at[2, "FeatureConstruction"] = (
            "LagsToBePlotted= %s" % DT_Setup_object.lags_4_plotting
        )
    DfMethodology.at[3, "FeatureConstruction"] = (
        "DifferenceCreate= %s" % DT_Setup_object.difference_create
    )
    if DT_Setup_object.difference_create == True:
        Word = (
            "All"
            if DT_Setup_object.create_differences == True
            else DT_Setup_object.create_differences
        )
        DfMethodology.at[4, "FeatureConstruction"] = (
            "FeaturesToCreateDifference= %s" % Word
        )
    DfMethodology.at[5, "FeatureConstruction"] = (
        "Manual creation of OwnLags= %s" % DT_Setup_object.create_manual_target_lag
    )
    if DT_Setup_object.create_manual_target_lag == True:
        DfMethodology.at[6, "FeatureConstruction"] = (
            "OwnLags= %s" % DT_Setup_object.target_lag
        )
    DfMethodology.at[7, "FeatureConstruction"] = (
        "Manual creation of FeatureLags= %s" % DT_Setup_object.create_manual_feature_lags
    )
    if DT_Setup_object.create_manual_feature_lags == True:
        DfMethodology.at[8, "FeatureConstruction"] = (
            "FeatureLags= %s" % DT_Setup_object.feature_lags
        )
    DfMethodology.at[9, "FeatureConstruction"] = (
        "Automatic creation of time series ownlags= %s"
        % DT_Setup_object.create_automatic_timeseries_target_lag
    )
    if DT_Setup_object.create_automatic_timeseries_target_lag == True:
        DfMethodology.at[10, "FeatureConstruction"] = (
            "Minimal Ownlag= %s" % DT_Setup_object.minimum_target_lag
        )
    DfMethodology.at[11, "FeatureConstruction"] = (
        "Automatic creation of lagged features= %s"
        % DT_Setup_object.create_automatic_feature_lags
    )
    if DT_Setup_object.create_automatic_feature_lags == True:
        DfMethodology.at[12, "FeatureConstruction"] = (
            "First lag to be considered= %s" % DT_Setup_object.minimum_feature_lag
        )
        DfMethodology.at[13, "FeatureConstruction"] = (
            "Last lag to be considered= %s" % DT_Setup_object.maximum_feature_lag
        )

    DfMethodology.at[1, "FeatureSelection"] = (
        "Manual feature selection = %s" % DT_Setup_object.manual_feature_selection
    )
    if DT_Setup_object.manual_feature_selection == True:
        DfMethodology.at[2, "FeatureSelection"] = (
            "Selected Features= %s" % DT_Setup_object.selected_features
        )
    DfMethodology.at[3, "FeatureSelection"] = (
        "Low Variance Filter = %s" % DT_Setup_object.low_variance_filter
    )
    if DT_Setup_object.low_variance_filter == True:
        DfMethodology.at[4, "FeatureSelection"] = (
            "Threshold Variance= %s" % DT_Setup_object.low_variance_threshold
        )
    DfMethodology.at[5, "FeatureSelection"] = (
        "Independent Component Analysis = %s" % DT_Setup_object.ICA
    )
    DfMethodology.at[6, "FeatureSelection"] = (
        "Univariate Filter = %s" % DT_Setup_object.univariate_filter
    )
    if DT_Setup_object.univariate_filter == True:
        DfMethodology.at[7, "FeatureSelection"] = (
            "Score function= %s" % DT_Setup_object.univariate_score_function
        )
        DfMethodology.at[8, "FeatureSelection"] = (
            "Search mode= %s" % DT_Setup_object.univariate_search_mode
        )
        DfMethodology.at[9, "FeatureSelection"] = (
            "Search mode threshold parameter= %s"
            % DT_Setup_object.univariate_filter_params
        )
    DfMethodology.at[
        10, "FeatureSelection"
    ] = "Embedded-Recursive Feature Selection = %s" % (
        DT_Setup_object.recursive_feature_selection
        or DT_Setup_object.embedded_feature_selection_threshold
    )
    if (
        DT_Setup_object.recursive_feature_selection
        or DT_Setup_object.embedded_feature_selection_threshold
    ) == True:
        DfMethodology.at[11, "FeatureSelection"] = (
            "Embedded Estimator = %s" % DT_Setup_object.embedded_model
        )
        if DT_Setup_object.recursive_feature_selection == True:
            DfMethodology.at[12, "FeatureSelection"] = (
                "Number of Features to select= %s"
                % DT_Setup_object.recursive_fs_number_features_to_select
            )
            if DT_Setup_object.recursive_fs_number_features_to_select == "automatic":
                DfMethodology.at[13, "FeatureSelection"] = (
                    "CrossValidation= %s" % DT_Setup_object.cross_validation_4_data_tuning
                )
            else:
                DfMethodology.at[14, "FeatureSelection"] = "CrossValidation= None"
        if DT_Setup_object.embedded_feature_selection_threshold == True:
            DfMethodology.at[15, "FeatureSelection"] = (
                "Feature importance threshold = %s" % DT_Setup_object.embedded_threshold_type
            )
            DfMethodology.at[16, "FeatureSelection"] = "CrossValidation= None"

    # save this dataframe in an excel
    ExcelFile = os.path.join(
        DT_Setup_object.abs_path_to_result_folder,
        "Settings_%s.xlsx" % (DT_Setup_object.name_of_data_tuning_experiment),
    )
    writer = pd.ExcelWriter(ExcelFile, engine="openpyxl")
    DfMethodology.to_excel(writer, sheet_name="Methodology")
    writer.save()
    writer.close()


def documentation_model_tuning(
    MT_Setup_Object,
    RR_Model_Summary,
    NameOfPredictor,
    Y_Predicted,
    Y_test,
    Y_train,
    ComputationTime,
    Scores,
    HyperparameterGrid=None,
    Bestparams=None,
    IndividualModel="",
    FeatureImportance="Not available",
):
    (R2, STD, RMSE, MAPE, MAE) = Scores

    # save summary of setup and evaluation
    dfSummary = pd.DataFrame(index=[0])
    dfSummary["Estimator"] = NameOfPredictor
    if Y_train is not None:  # don´t document this if "onlypredict" is used
        dfSummary["Start_date_Fit"] = MT_Setup_Object.StartTraining
        dfSummary["End_date_Fit"] = MT_Setup_Object.EndTraining
    dfSummary["Start_date_Predict"] = MT_Setup_Object.StartTesting
    dfSummary["End_date_Predict"] = MT_Setup_Object.EndTesting
    if Y_train is not None:  # don´t document this if "onlypredict" is used
        dfSummary["Total Train Samples"] = len(Y_train.index)
    dfSummary["Test Samples"] = len(Y_test.index)
    dfSummary["Recursive"] = MT_Setup_Object.GlobalRecu
    dfSummary["Shuffle"] = MT_Setup_Object.GlobalShuffle
    if HyperparameterGrid is not None:
        dfSummary["Range Hyperparameter"] = str(HyperparameterGrid)
        dfSummary["CrossValidation"] = str(MT_Setup_Object.GlobalCV_MT)
        dfSummary["Best Hyperparameter"] = str(Bestparams)
        if MT_Setup_Object.GlobalMaxEval_HyParaTuning is not None:
            dfSummary["Max Bayesian Evaluations"] = str(
                MT_Setup_Object.GlobalMaxEval_HyParaTuning
            )
    dfSummary["Feature importance"] = str(FeatureImportance)
    dfSummary["Individual model"] = IndividualModel
    if IndividualModel == "byFeature":
        dfSummary["IndivFeature"] = MT_Setup_Object.IndivFeature
        dfSummary["IndivThreshold"] = MT_Setup_Object.IndivThreshold
    dfSummary["Eval_R2"] = R2
    dfSummary["Eval_RMSE"] = RMSE
    dfSummary["Eval_MAPE"] = MAPE
    dfSummary["Eval_MAE"] = MAE
    dfSummary["Standard deviation"] = STD
    dfSummary["Computation Time"] = "%.2f seconds" % ComputationTime
    dfSummary = dfSummary.T
    # write summary of setup and evaluation in excel File
    SummaryFile = os.path.join(
        MT_Setup_Object.ResultsFolderSubTest,
        "Summary_%s_%s.xlsx" % (NameOfPredictor, MT_Setup_Object.NameOfSubTest),
    )
    writer = pd.ExcelWriter(SummaryFile)
    dfSummary.to_excel(writer, float_format="%.6f")
    writer.save()

    # export prediction to Excel
    SaveFileName_excel = os.path.join(
        MT_Setup_Object.ResultsFolderSubTest,
        "Prediction_%s_%s.xlsx" % (NameOfPredictor, MT_Setup_Object.NameOfSubTest),
    )
    Y_Predicted.to_frame(name=MT_Setup_Object.name_of_target).to_excel(SaveFileName_excel)

    # save model tuning runtime results in ModelTuningRuntimeResults class object

    RR_Model_Summary.model_name = NameOfPredictor
    if Y_train is not None:
        RR_Model_Summary.total_train_samples = len(Y_train.index)
    RR_Model_Summary.test_samples = len(Y_test.index)
    if HyperparameterGrid is not None:
        RR_Model_Summary.best_hyperparameter = str(Bestparams)
    RR_Model_Summary.feature_importance = str(FeatureImportance)
    RR_Model_Summary.eval_R2 = R2
    RR_Model_Summary.eval_RMSE = RMSE
    RR_Model_Summary.eval_MAPE = MAPE
    RR_Model_Summary.eval_MAE = MAE
    RR_Model_Summary.standard_deviation = STD
    RR_Model_Summary.computation_time = "%.2f seconds" % ComputationTime

    # return Score for modelselection
    return R2


def documentation_only_predict(
    MT_Setup_Object_PO,
    RR_Model_Summary,
    NameOfPredictor,
    Y_Predicted,
    Y_test,
    ComputationTime,
    Scores,
    IndividualModel="",
    FeatureImportance="Not available",
):
    (R2, STD, RMSE, MAPE, MAE) = Scores

    # save summary of setup and evaluation
    dfSummary = pd.DataFrame(index=[0])
    dfSummary["Estimator"] = NameOfPredictor
    dfSummary["Start_date_Predict"] = MT_Setup_Object_PO.StartTesting
    dfSummary["End_date_Predict"] = MT_Setup_Object_PO.EndTesting
    dfSummary["Test Samples"] = len(Y_test.index)
    dfSummary["Recursive"] = MT_Setup_Object_PO.OnlyPredictRecursive
    dfSummary["Shuffle"] = None
    dfSummary["Feature importance"] = str(FeatureImportance)
    dfSummary["Individual model"] = IndividualModel
    if IndividualModel == "byFeature":
        dfSummary["IndivFeature"] = MT_Setup_Object_PO.IndivFeature
        dfSummary["IndivThreshold"] = MT_Setup_Object_PO.IndivThreshold
    dfSummary["Eval_R2"] = R2
    dfSummary["Eval_RMSE"] = RMSE
    dfSummary["Eval_MAPE"] = MAPE
    dfSummary["Eval_MAE"] = MAE
    dfSummary["Standard deviation"] = STD
    dfSummary["Computation Time"] = "%.2f seconds" % ComputationTime
    dfSummary = dfSummary.T
    # write summary of setup and evaluation in excel File
    SummaryFile = os.path.join(
        MT_Setup_Object_PO.OnlyPredictFolder,
        "Summary_%s_%s.xlsx" % (NameOfPredictor, MT_Setup_Object_PO.NameOfSubTest),
    )
    writer = pd.ExcelWriter(SummaryFile)
    dfSummary.to_excel(writer, float_format="%.6f")
    writer.save()

    # export prediction to Excel
    SaveFileName_excel = os.path.join(
        MT_Setup_Object_PO.OnlyPredictFolder,
        "Prediction_%s_%s.xlsx" % (NameOfPredictor, MT_Setup_Object_PO.NameOfSubTest),
    )
    Y_Predicted.to_frame(name=MT_Setup_Object_PO.name_of_target).to_excel(
        SaveFileName_excel
    )

    # save model tuning runtime results in ModelTuningRuntimeResults class object

    RR_Model_Summary.model_name = NameOfPredictor
    RR_Model_Summary.test_samples = len(Y_test.index)
    RR_Model_Summary.feature_importance = str(FeatureImportance)
    RR_Model_Summary.eval_R2 = R2
    RR_Model_Summary.eval_RMSE = RMSE
    RR_Model_Summary.eval_MAPE = MAPE
    RR_Model_Summary.eval_MAE = MAE
    RR_Model_Summary.standard_deviation = STD
    RR_Model_Summary.computation_time = "%.2f seconds" % ComputationTime


def visualization(MT_Setup_Object, NameOfPredictor, prediction, measurement, Scores):
    (R2, STD, RMSE, MAPE, MAE) = Scores

    plot_predict_measured(
        prediction,
        measurement,
        MAE=MAE,
        R2=R2,
        StartDatePredict=MT_Setup_Object.StartTesting,
        SavePath=MT_Setup_Object.ResultsFolderSubTest,
        nameOfSignal=MT_Setup_Object.name_of_target,
        BlackBox=NameOfPredictor,
        NameOfSubTest=MT_Setup_Object.NameOfSubTest,
    )

    plot_Residues(
        prediction,
        measurement,
        MAE=MAE,
        R2=R2,
        SavePath=MT_Setup_Object.ResultsFolderSubTest,
        nameOfSignal=MT_Setup_Object.name_of_target,
        BlackBox=NameOfPredictor,
        NameOfSubTest=MT_Setup_Object.NameOfSubTest,
    )


# OP
def documentation_iterative_evaluation(
    MT_Setup_object_PO,
    NameOfPredictor,
    mean_score,
    SD_score,
    errorlist,
    horizon,
    errormetric,
):
    errorlist = np.around(errorlist, 3)
    # save results of iterative evaluation in the summary file
    ExcelFile = os.path.join(
        MT_Setup_object_PO.OnlyPredictFolder,
        "Summary_%s_%s.xlsx" % (NameOfPredictor, MT_Setup_object_PO.NameOfSubTest),
    )
    Excel = pd.read_excel(ExcelFile)
    book = load_workbook(ExcelFile)
    writer = pd.ExcelWriter(ExcelFile, engine="openpyxl")
    writer.book = book
    writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
    # create dataframe containing the information
    ErrorDF = pd.DataFrame(index=[0])
    ErrorDF["________"] = "_________________________________"
    if MT_Setup_object_PO.ValidationPeriod == True:
        ErrorDF["Test Data"] = (
            "Interpretation of error measures of the data from %s till %s, per error metric"
            % (
                MT_Setup_object_PO.StartTest_onlypredict,
                MT_Setup_object_PO.EndTest_onlypredict,
            )
        )
    else:
        ErrorDF[
            "Test Data"
        ] = "Interpretation of error measures regarding the whole data set per error metric"
    ErrorDF["Used error metric"] = str(errormetric)
    ErrorDF["Horizon length"] = horizon
    ErrorDF["Mean score"] = "%.3f" % mean_score
    ErrorDF["Standard deviation of errors"] = SD_score
    ErrorDF["Max score"] = str(max(errorlist))
    ErrorDF["Min score"] = str(min(errorlist))
    ErrorDF["Number of tested folds"] = len(errorlist)
    ErrorDF = ErrorDF.T

    ErrorListDF = pd.DataFrame(index=range(len(errorlist)))
    ErrorListDF["List of errors"] = errorlist
    ErrorListDF = ErrorListDF.T

    Excel = pd.concat([Excel, ErrorDF, ErrorListDF])

    Excel.to_excel(writer, sheet_name="Sheet1")
    writer.save()
    writer.close()

'''This Module contains classes for storing runtime results'''

import os

from sklearn.externals import joblib

class RRSummary:
    '''
    Stores runtime results for one algorithm.
    '''

    def __init__(self, **kwargs):
        self.model_name = "model"
        self.total_train_samples = 0
        self.test_samples = 0
        self.best_hyperparameter = None
        self.feature_importance = None
        self.eval_R2 = 0
        self.eval_RMSE = 0
        self.eval_MAPE = 0
        self.eval_MAE = 0
        self.standard_deviation = 0
        self.computation_time = 0

class ModelTuningRuntimeResults:
    """
    Stores all runtime results of ModelTuning.py, AutoFinalBayes.py, OnlyPredict.py per used algorithm.


    Stores the following summary:

    1. Total Train Samples
    2. Test Samples
    3. Best Hyperparameter
    4. Feature importance
    5. Eval_R2
    6. Eval_RMSE
    7. Eval_MAPE
    8. Eval_MAE
    9. Standard deviation
    10.Computation Time

    """

    def __init__(self, **kwargs):
        self.SVR_Summary = RRSummary()
        self.RF_Summary = RRSummary()
        self.ANN_Summary = RRSummary()
        self.GB_Summary = RRSummary()
        self.Lasso_Summary = RRSummary()
        self.ModelSelection_Summary = RRSummary()
        self.SVR_grid_Summary = RRSummary()
        self.ANN_grid_Summary = RRSummary()
        self.GB_grid_Summary = RRSummary()
        self.Lasso_grid_Summary = RRSummary()

    def store_results(self, MT_Setup_object):
        print("Saving Model Tuning Runtime Results class Object as a pickle in path: '%s'" % os.path.join(
            MT_Setup_object.ResultsFolder, "DataTuningRuntimeResults.save"))

        # Save the object as a pickle for reuse
        joblib.dump(self, os.path.join(MT_Setup_object.ResultsFolder, "ModelTuningRuntimeResults.save"))

        return

import os

import pandas as pd
from sklearn.externals import joblib

class ModelTuningRuntimeResults:
    """
    Object that stores all the runtime results of Model Tuning

    Stores all the data frames containing the predictions.
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

    def store_results(self, DT_Setup_object):
        print("Saving Data Tuning Runtime Results class Object as a pickle in path: '%s'" % os.path.join(DT_Setup_object.ResultsFolder, "DataTuningRuntimeResults.save"))

        # Save the object as a pickle for reuse
        joblib.dump(self, os.path.join(DT_Setup_object.ResultsFolder, "DataTuningRuntimeResults.save"))
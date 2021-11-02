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
        # 1 ANN Bayesian Predictor
        self.ann_total_train_samples = 0
        self.ann_test_samples = 0
        self.ann_best_hyperparameter = None
        self.ann_feature_importance = None
        self.ann_eval_R2 = 0
        self.ann_eval_RMSE = 0
        self.ann_eval_MAPE = 0
        self.ann_eval_MAE = 0
        self.ann_standard_deviation = 0
        self.ann_computation_time = 0

        # 2 ANN Grid Search Predictor
        self.ann_grid_total_train_samples = 0
        self.ann_grid_test_samples = 0
        self.ann_grid_best_hyperparameter = None
        self.ann_grid_feature_importance = None
        self.ann_grid_eval_R2 = 0
        self.ann_grid_eval_RMSE = 0
        self.ann_grid_eval_MAPE = 0
        self.ann_grid_eval_MAE = 0
        self.ann_grid_standard_deviation = 0
        self.ann_grid_computation_time = 0

        # 3 Gradient Boost Bayesian Predictor
        self.gradient_total_train_samples = 0
        self.gradient_test_samples = 0
        self.gradient_best_hyperparameter = None
        self.gradient_feature_importance = None
        self.gradient_eval_R2 = 0
        self.gradient_eval_RMSE = 0
        self.gradient_eval_MAPE = 0
        self.gradient_eval_MAE = 0
        self.gradient_standard_deviation = 0
        self.gradient_computation_time = 0

        # 4 Gradient Boost Grid Search Predictor
        self.gradient_grid_total_train_samples = 0
        self.gradient_grid_test_samples = 0
        self.gradient_grid_best_hyperparameter = None
        self.gradient_grid_feature_importance = None
        self.gradient_grid_eval_R2 = 0
        self.gradient_grid_eval_RMSE = 0
        self.gradient_grid_eval_MAPE = 0
        self.gradient_grid_eval_MAE = 0
        self.gradient_grid_standard_deviation = 0
        self.gradient_grid_computation_time = 0

        # 5 Lasso Bayesian Predictor
        self.lasso_total_train_samples = 0
        self.lasso_test_samples = 0
        self.lasso_best_hyperparameter = None
        self.lasso_feature_importance = None
        self.lasso_eval_R2 = 0
        self.lasso_eval_RMSE = 0
        self.lasso_eval_MAPE = 0
        self.lasso_eval_MAE = 0
        self.lasso_standard_deviation = 0
        self.lasso_computation_time = 0

        # 6 Lasso Gradient Search Predictor
        self.lasso_grid_total_train_samples = 0
        self.lasso_grid_test_samples = 0
        self.lasso_grid_best_hyperparameter = None
        self.lasso_grid_feature_importance = None
        self.lasso_grid_eval_R2 = 0
        self.lasso_grid_eval_RMSE = 0
        self.lasso_grid_eval_MAPE = 0
        self.lasso_grid_eval_MAE = 0
        self.lasso_grid_standard_deviation = 0
        self.lasso_grid_computation_time = 0

        # 7 RF Predictor
        self.rf_total_train_samples = 0
        self.rf_test_samples = 0
        self.rf_best_hyperparameter = None
        self.rf_feature_importance = None
        self.rf_eval_R2 = 0
        self.rf_eval_RMSE = 0
        self.rf_eval_MAPE = 0
        self.rf_eval_MAE = 0
        self.rf_standard_deviation = 0
        self.rf_computation_time = 0

        # 8 SVR Bayesian Predictor
        self.svr_total_train_samples = 0
        self.svr_test_samples = 0
        self.svr_best_hyperparameter = None
        self.svr_feature_importance = None
        self.svr_eval_R2 = 0
        self.svr_eval_RMSE = 0
        self.svr_eval_MAPE = 0
        self.svr_eval_MAE = 0
        self.svr_standard_deviation = 0
        self.svr_computation_time = 0

        # 9 SVR Grid Search Predictor
        self.svr_grid_total_train_samples = 0
        self.svr_grid_test_samples = 0
        self.svr_grid_best_hyperparameter = None
        self.svr_grid_feature_importance = None
        self.svr_grid_eval_R2 = 0
        self.svr_grid_eval_RMSE = 0
        self.svr_grid_eval_MAPE = 0
        self.svr_grid_eval_MAE = 0
        self.svr_grid_standard_deviation = 0
        self.svr_grid_computation_time = 0

    def store_results(self, MT_Setup_object):
        print("Saving Model Tuning Runtime Results class Object as a pickle in path: '%s'" % os.path.join(
            MT_Setup_object.ResultsFolder, "DataTuningRuntimeResults.save"))

        # Save the object as a pickle for reuse
        joblib.dump(self, os.path.join(MT_Setup_object.ResultsFolder, "ModelTuningRuntimeResults.save"))

        return

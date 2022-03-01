"""This Module contains classes for storing runtime results"""

import os
from sklearn.externals import joblib


class RRSummary:
    """
    Stores runtime results for one algorithm.
    """

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
        self.SVR_grid_Summary = RRSummary()
        self.ANN_grid_Summary = RRSummary()
        self.GB_grid_Summary = RRSummary()
        self.Lasso_grid_Summary = RRSummary()
        self.ModelSelection_Summary = RRSummary()

    def store_results(self, MT_Setup_object):
        print(
            "Saving Model Tuning Runtime Results class Object as a pickle in path: \n'%s'"
            % os.path.join(
                MT_Setup_object.ResultsFolder, "DataTuningRuntimeResults.save"
            )
        )

        # Save the object as a pickle for reuse
        joblib.dump(
            self,
            os.path.join(
                MT_Setup_object.ResultsFolder, "ModelTuningRuntimeResults.save"
            ),
        )

        return


# saves the BestModels in a folder "BestModels", also capable of saving individual models
def model_saver(Result_dic, ResultsFolderSubTest, NameOfPredictor, IndividualModel):
    if os.path.isdir(os.path.join(ResultsFolderSubTest, "BestModels")) == True:
        pass
    else:
        os.makedirs(os.path.join(ResultsFolderSubTest, "BestModels"))

    if IndividualModel == "week_weekend":
        joblib.dump(
            Result_dic["Best_trained_model"]["weekday"],
            os.path.join(
                ResultsFolderSubTest,
                "BestModels",
                "weekday_%s.save" % (NameOfPredictor),
            ),
        )  # dump the best trained model in a file to reuse it for different predictions
        joblib.dump(
            Result_dic["Best_trained_model"]["weekend"],
            os.path.join(
                ResultsFolderSubTest,
                "BestModels",
                "weekend_%s.save" % (NameOfPredictor),
            ),
        )  # dump the best trained model in a file to reuse it for different predictions
    elif IndividualModel == "hourly":
        joblib.dump(
            Result_dic["Best_trained_model"][0],
            os.path.join(
                ResultsFolderSubTest, "BestModels", "0_%s.save" % (NameOfPredictor)
            ),
        )
        joblib.dump(
            Result_dic["Best_trained_model"][1],
            os.path.join(
                ResultsFolderSubTest, "BestModels", "1_%s.save" % (NameOfPredictor)
            ),
        )
        joblib.dump(
            Result_dic["Best_trained_model"][2],
            os.path.join(
                ResultsFolderSubTest, "BestModels", "2_%s.save" % (NameOfPredictor)
            ),
        )
        joblib.dump(
            Result_dic["Best_trained_model"][3],
            os.path.join(
                ResultsFolderSubTest, "BestModels", "3_%s.save" % (NameOfPredictor)
            ),
        )
        joblib.dump(
            Result_dic["Best_trained_model"][4],
            os.path.join(
                ResultsFolderSubTest, "BestModels", "4_%s.save" % (NameOfPredictor)
            ),
        )
        joblib.dump(
            Result_dic["Best_trained_model"][5],
            os.path.join(
                ResultsFolderSubTest, "BestModels", "5_%s.save" % (NameOfPredictor)
            ),
        )
        joblib.dump(
            Result_dic["Best_trained_model"][6],
            os.path.join(
                ResultsFolderSubTest, "BestModels", "6_%s.save" % (NameOfPredictor)
            ),
        )
        joblib.dump(
            Result_dic["Best_trained_model"][7],
            os.path.join(
                ResultsFolderSubTest, "BestModels", "7_%s.save" % (NameOfPredictor)
            ),
        )
        joblib.dump(
            Result_dic["Best_trained_model"][8],
            os.path.join(
                ResultsFolderSubTest, "BestModels", "8_%s.save" % (NameOfPredictor)
            ),
        )
        joblib.dump(
            Result_dic["Best_trained_model"][9],
            os.path.join(
                ResultsFolderSubTest, "BestModels", "9_%s.save" % (NameOfPredictor)
            ),
        )
        joblib.dump(
            Result_dic["Best_trained_model"][10],
            os.path.join(
                ResultsFolderSubTest, "BestModels", "10_%s.save" % (NameOfPredictor)
            ),
        )
        joblib.dump(
            Result_dic["Best_trained_model"][11],
            os.path.join(
                ResultsFolderSubTest, "BestModels", "11_%s.save" % (NameOfPredictor)
            ),
        )
        joblib.dump(
            Result_dic["Best_trained_model"][12],
            os.path.join(
                ResultsFolderSubTest, "BestModels", "12_%s.save" % (NameOfPredictor)
            ),
        )
        joblib.dump(
            Result_dic["Best_trained_model"][13],
            os.path.join(
                ResultsFolderSubTest, "BestModels", "13_%s.save" % (NameOfPredictor)
            ),
        )
        joblib.dump(
            Result_dic["Best_trained_model"][14],
            os.path.join(
                ResultsFolderSubTest, "BestModels", "14_%s.save" % (NameOfPredictor)
            ),
        )
        joblib.dump(
            Result_dic["Best_trained_model"][15],
            os.path.join(
                ResultsFolderSubTest, "BestModels", "15_%s.save" % (NameOfPredictor)
            ),
        )
        joblib.dump(
            Result_dic["Best_trained_model"][16],
            os.path.join(
                ResultsFolderSubTest, "BestModels", "16_%s.save" % (NameOfPredictor)
            ),
        )
        joblib.dump(
            Result_dic["Best_trained_model"][17],
            os.path.join(
                ResultsFolderSubTest, "BestModels", "17_%s.save" % (NameOfPredictor)
            ),
        )
        joblib.dump(
            Result_dic["Best_trained_model"][18],
            os.path.join(
                ResultsFolderSubTest, "BestModels", "18_%s.save" % (NameOfPredictor)
            ),
        )
        joblib.dump(
            Result_dic["Best_trained_model"][19],
            os.path.join(
                ResultsFolderSubTest, "BestModels", "19_%s.save" % (NameOfPredictor)
            ),
        )
        joblib.dump(
            Result_dic["Best_trained_model"][20],
            os.path.join(
                ResultsFolderSubTest, "BestModels", "20_%s.save" % (NameOfPredictor)
            ),
        )
        joblib.dump(
            Result_dic["Best_trained_model"][21],
            os.path.join(
                ResultsFolderSubTest, "BestModels", "21_%s.save" % (NameOfPredictor)
            ),
        )
        joblib.dump(
            Result_dic["Best_trained_model"][22],
            os.path.join(
                ResultsFolderSubTest, "BestModels", "22_%s.save" % (NameOfPredictor)
            ),
        )
        joblib.dump(
            Result_dic["Best_trained_model"][23],
            os.path.join(
                ResultsFolderSubTest, "BestModels", "23_%s.save" % (NameOfPredictor)
            ),
        )
    elif IndividualModel == "byFeature":
        joblib.dump(
            Result_dic["Best_trained_model"]["above"],
            os.path.join(
                ResultsFolderSubTest, "BestModels", "above_%s.save" % (NameOfPredictor)
            ),
        )
        joblib.dump(
            Result_dic["Best_trained_model"]["below"],
            os.path.join(
                ResultsFolderSubTest, "BestModels", "below_%s.save" % (NameOfPredictor)
            ),
        )
    else:
        joblib.dump(
            Result_dic["Best_trained_model"],
            os.path.join(
                ResultsFolderSubTest, "BestModels", "%s.save" % (NameOfPredictor)
            ),
        )

import pandas as pd
import numpy as np
import sklearn
import os
import tensorflow
import keras
import scikeras
from core.s3_model_tuning.models.scikit_learn_models import BaseScikitLearnModel
from core.s3_model_tuning.models.scikit_learn_models import ScikitMLP, LinearReg
from core.s3_model_tuning.models.model_factory import ModelFactory
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from core.s3_model_tuning.models.abstract_model import AbstractMLModel
from core.s3_model_tuning.models.keras_models import BaseKerasModel, SciKerasSequential
from core.util.definitions import root_dir
from core.s3_model_tuning.config.model_tuning_config import ModelTunerConfig
from core.executables.exe_model_tuning import exe_model_tuning
from core.s3_model_tuning.model_tuner import ModelTuner

from extrapolation_detection.util import loading_saving


    # Load and prepare data
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['price'] = pd.Series(data.target)
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df['price'], test_size=0.2,
                                                                            random_state=42)
dir_path = os.path.join(root_dir(), "0000_testfiles")

model = SciKerasSequential()
model.get_params()
model.fit(X_train, y_train)
model.save_regressor(dir_path, "regressor", "h5")
model1 = ModelFactory.load_model(r"D:\PyCharm 2023.3.5\pythonProject\addmo-automated-ml-regression\0000_testfiles\regressor.h5")
print(model1)
y_pred1= model1.predict(X_test)
y_pred = model.predict(X_test)
r1= r2_score(y_test, y_pred)
r2= r2_score(y_test, y_pred1 )
print("model loaded: ", r1, "model: ", r2)

# print(y_pred)
# print(keras.__version__)
# print(tensorflow.__version__)
# print(scikeras.__version__)
# def setup_model_tuning_config():
#     # Configures and returns a ModelTunerConfig instance
#     return ModelTunerConfig(
#         models=["SciKerasSequential"],
#         hyperparameter_tuning_type="OptunaTuner",
#         hyperparameter_tuning_kwargs={"n_trials": 3},
#         validation_score_mechanism="cv",
#         validation_score_splitting="KFold",
#         validation_score_metric="neg_mean_squared_error"
#     )
#
# config = setup_model_tuning_config()
# tuner = ModelTuner(config)
# model_dict = tuner.tune_all_models(X_train, y_train)
# print(model_dict)
# best_model_name = tuner.get_best_model_name(model_dict)
# best_model = tuner.get_model(model_dict, "SciKerasSequential")
# print("best model name is: ", best_model_name)
# print("best model is: ", best_model)
# regressor=best_model
# p= regressor.get_params()
# print(p)
# y_pred= regressor.predict(X_test)
# r_score= r2_score(y_test, y_pred)
# print(r_score)

import pandas as pd
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense
from core.s3_model_tuning.models.scikit_learn_models import BaseScikitLearnModel
from core.s3_model_tuning.models.scikit_learn_models import ScikitMLP, LinearReg
from core.s3_model_tuning.models.model_factory import ModelFactory
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from core.s3_model_tuning.models.abstract_model import AbstractMLModel
from core.s3_model_tuning.models.keras_model import BaseKerasModel, SciKerasSequential
from core.util.definitions import root_dir
from core.s3_model_tuning.config.model_tuning_config import ModelTunerConfig
from core.executables.exe_model_tuning import exe_model_tuning
from core.s3_model_tuning.model_tuner import ModelTuner
from scikeras.wrappers import KerasRegressor


data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['price'] = pd.Series(data.target)
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df['price'], test_size=0.2, random_state=42)
dir_path = os.path.join(root_dir(), "0000_testfiles")

# Trial example:
def build_model():
    model = Sequential([
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model# Instantiate ScikerasRegressor with the default Keras model
scikeras_regressor = KerasRegressor(build_fn=build_model)# Example training data
X_train = np.random.rand(100, 10)  # Example input data with shape (100, 10)
y_train = np.random.rand(100)      # Example target data with shape (100,)# Fit the model for the first time, allowing the model to infer input shape
scikeras_regressor.fit(X_train, y_train)# Example test data
X_test = np.random.rand(10, 10)    # Example input data for prediction with shape (10, 10)# Make predictions
predictions = scikeras_regressor.predict(X_test)
#
#
# model= SciKerasSequential()
# p=model.get_params()
# print(p)
# model.fit(X_train,y_train)
#
# model.save_regressor(dir_path, file_type='h5')
#
#     # Load the model
# loaded_model = ModelFactory().load_model(os.path.join(dir_path, f"{type(model).__name__}.{'h5'}"))
#
#     # Make predictions
# y_pred_loaded = loaded_model.predict(X_test)
#
#     # Calculate R-squared
# r_squared_loaded = r2_score(y_test, y_pred_loaded)
# print(r_squared_loaded)
#
#
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
#
# config = setup_model_tuning_config()
# tuner = ModelTuner(config)
# model_dict = tuner.tune_all_models(x_train_val=X_train, y_train_val=y_train)
# best_model_name = tuner.get_best_model_name(model_dict)
# best_model = tuner.get_model(model_dict, best_model_name)
# score = tuner.get_model_validation_score(model_dict, best_model_name)
#
# print('Best Model is: ', best_model_name, best_model)
# print('Score is: ', score)
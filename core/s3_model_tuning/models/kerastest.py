import pandas as pd
import numpy as np
import os
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




    # Load and prepare data
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['price'] = pd.Series(data.target)
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df['price'], test_size=0.2,
                                                                            random_state=42)
dir_path = os.path.join(root_dir(), "0000_testfiles")

model = SciKerasSequential()
model.fit(X_train, y_train)

# Testing saving and loading of model
model.save_regressor(dir_path, file_type='h5')

    # Load the model
loaded_model = ModelFactory().load_model(os.path.join(dir_path, f"{type(model).__name__}.{'h5'}"))
print(loaded_model)
    # Make predictions
y_pred_loaded = loaded_model.predict(X_test)

    # Calculate R-squared
r_squared_loaded = r2_score(y_test, y_pred_loaded)


def setup_model_tuning_config():
    # Configures and returns a ModelTunerConfig instance
    return ModelTunerConfig(
        models=["SciKerasSequential"],
        hyperparameter_tuning_type="OptunaTuner",
        hyperparameter_tuning_kwargs={"n_trials": 3},
        validation_score_mechanism="cv",
        validation_score_splitting="KFold",
        validation_score_metric="neg_mean_squared_error"
    )

config = setup_model_tuning_config()
tuner = ModelTuner(config)
model = tuner.tune_model('SciKerasSequential',x_train_val=X_train, y_train_val=y_train)
# best_model_name = tuner.get_best_model_name(model_dict)
# best_model = tuner.get_model(model_dict, best_model_name)
# score = tuner.get_model_validation_score(model_dict, best_model_name)
print(model)
# print('Best Model is: ', best_model_name, best_model)
# print('Score is: ', score)
# p= best_model.get_params()
# print(p)
# regressor = best_model
#
#     # generate prediction for fit period
# regressor.fit(X_train, y_train)
# y_pred = model.predict(X_train)

y_pred = pd.Series(model.predict(X_train).ravel())
print(y_pred)
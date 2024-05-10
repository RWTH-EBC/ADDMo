import unittest
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
from core.s3_model_tuning.models.keras_model import BaseKerasModel
from core.util.definitions import root_dir
from core.s3_model_tuning.hyperparameter_tuning.hyperparameter_tuner import OptunaTuner
from core.s3_model_tuning.config.model_tuning_config import ModelTuningExperimentConfig
from core.s3_model_tuning.scoring.abstract_scorer import ValidationScoring


data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['price'] = pd.Series(data.target)
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df['price'], test_size=0.2, random_state=42)
dir_path = os.path.join(root_dir(), "0000_testfiles")

model = BaseKerasModel()
model.fit(X_train, y_train)

model.save_regressor(dir_path, file_type="keras")

    # Load the model
loaded_model = ModelFactory().load_model(os.path.join(dir_path, f"{type(model).__name__}.{'keras'}"))

    # Make predictions
y_pred_loaded = loaded_model.predict(X_test)

    # Calculate R-squared
r_squared_loaded = r2_score(y_test, y_pred_loaded)

sk=loaded_model.to_scikit_learn()
p= sk.get_params()
print("The current parameters are: ", p)
params= {"loss": "rmsprop", "batch_size": 64}
sk.set_params(**params)
p=sk.get_params()
print(p)
config = ModelTuningExperimentConfig(n_trials=10)
scorer = ValidationScoring(metric='mean_squared_error')
tuner = OptunaTuner(config, scorer)  # Create an instance of the OptunaTuner
best_params = tuner.tune(model=sk, x_train_val=X_train, y_train_val=y_train)
print("Best hyperparameters:", best_params)

# Set the best hyperparameters
sk.set_params(**best_params)
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
from core.s3_model_tuning.scoring.abstract_scorer import Scoring
from core.s3_model_tuning.config.model_tuning_config import ModelTunerConfig
from core.executables.exe_model_tuning import exe_model_tuning
from core.s3_model_tuning.model_tuner import ModelTuner

data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df.head()
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


def setup_model_tuning_config() -> ModelTunerConfig:
    config_model_tuner = ModelTunerConfig(
        models=["BaseKerasModel"],
        hyperparameter_tuning_type="OptunaTuner",
        hyperparameter_tuning_kwargs={"n_trials": 10},  # Adjust the number of trials as needed
        validation_score_mechanism="cv",
        validation_score_splitting="KFold",
        validation_score_metric="neg_mean_squared_error"
    )

    return config_model_tuner

config= setup_model_tuning_config()

tuner = ModelTuner(config)
model_dict = tuner.tune_model(model_name='BaseKerasModel', x_train_val=X_train, y_train_val=y_train)
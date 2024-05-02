import os

import joblib
import json
import sklearn

import pandas as pd
from abc import ABC
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from skl2onnx import to_onnx
from core.s3_model_tuning.models.abstract_model import AbstractMLModel
from core.s3_model_tuning.models.abstract_model import ModelMetadata
from sklearn.linear_model import LinearRegression
from core.util.definitions import get_commit_id


class BaseScikitLearnModel(AbstractMLModel, ABC):
    """
    Base class for scikit-learn models.
    This class extends the AbstractMLModel, providing concrete implementations of
    common functionalities specific to scikit-learn models.

    Attributes:
        model (Pipeline): A scikit-learn Pipeline object containing the scaler and the provided model.
    """

    def __init__(self, regressor):
        # Create an instance of the scikit-learn model including a scaler
        self.regressor = Pipeline(
            steps=[
                ("scaler", StandardScaler()),  # scale the features
                ("model", regressor)  # scaling the target variable through TransformedTargetRegressor
                # is not compatible with ONNX
            ]
        )

    def fit(self, x, y):  # Todo catch exception if x or y is not a pandas dataframe / update comments
        self.x = x  # Save the training data to be used later for ONNX conversion
        self.y = y  # Save the target column to get target name for metadata
        self.regressor.fit(x, y)  # Train the model

    def predict(self, x):
        return self.regressor.predict(x)  # Make predictions

    def _save_metadata(self, directory, regressor_filename):

        # feature names can only be extracted if x and y are pandas dataframes
        if not isinstance(self.x, pd.DataFrame):
            raise TypeError('x should be a pandas dataframe and not of type', type(self.x))
        if not isinstance(self.y, (pd.Series, pd.DataFrame)):
            raise TypeError('y should be a pandas dataframe/series and not of type', type(self.y))

        # define metadata
        self.metadata = ModelMetadata(
            addmo_class=type(self).__name__,
            addmo_commit_id=get_commit_id(),
            library=sklearn.__name__,
            library_model_type=type(self.regressor.named_steps['model']).__name__,
            library_version=sklearn.__version__,
            target_name=self.y.name,
            features_ordered=list(self.x.columns),
            preprocessing=['StandardScaler for all features'])

        # save metadata
        regressor_filename = os.path.splitext(regressor_filename)[0]
        metadata_path = os.path.join(directory, regressor_filename + '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata.dict(), f)

    def save_regressor(self, directory, filename=None, file_type='joblib'):

        if filename is None:
            filename = type(self).__name__

        path = os.path.join(directory, f"{filename}.{file_type}")

        if file_type == 'joblib':
            joblib.dump(self.regressor, path)
            self._save_metadata(directory, filename)
            print(f"Model saved to {path}")

        elif file_type == 'onnx':
            onx = to_onnx(self.regressor, self.x[:1])
            self._save_metadata(directory, filename)
            with open(path, "wb") as f:
                f.write(onx.SerializeToString())
                print(f"Model saved to {path}.")

    def load_regressor(self, regressor):
        self.regressor = regressor

    def to_scikit_learn(self):
        return self.regressor

    def set_params(self, hyperparameters):
        # access the hyperparameters of the model within the pipeline within the
        # TransformedTargetRegressor
        self.regressor.named_steps["model"].set_params(**hyperparameters)

    def get_params(self, deep=True):
        # Get the hyperparameters of the model
        return self.regressor.named_steps["model"].get_params(deep=deep)

    def default_hyperparameter(self):
        return self.regressor.get_params()
    def build_regressor(self):
        pass


class ScikitMLP(BaseScikitLearnModel):
    """Scikit-learn MLPRegressor model."""

    def __init__(self):
        super().__init__(MLPRegressor())

    def optuna_hyperparameter_suggest(self, trial):
        hyperparameters = {}

        # Suggest hyperparameters
        n_layers = trial.suggest_int("n_layers", 1, 3)
        hidden_layer_sizes = tuple(
            trial.suggest_int(f"n_units_l{i}", 1, 100) for i in range(n_layers)
        )

        # Dynamic hidden layer sizes based on the number of layers
        hyperparameters["hidden_layer_sizes"] = hidden_layer_sizes

        # Other hyperparameters
        hyperparameters["activation"] = "relu"
        hyperparameters["max_iter"] = 2000

        return hyperparameters

    def grid_search_hyperparameter(self):
        hyperparameter_grid = {
            "hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 100)],
            "activation": ["tanh", "relu"],
            "solver": ["sgd", "adam"],
            "alpha": [0.0001, 0.05],
            "learning_rate": ["constant", "adaptive"],
        }
        return hyperparameter_grid


class ScikitMLP_TargetTransformed(ScikitMLP):
    def __init__(self):
        # Create an instance of the scikit-learn model including a scaler
        self.regressor = Pipeline(
            steps=[
                ("scaler", StandardScaler()),  # scale the features
                ("model", TransformedTargetRegressor(regressor=MLPRegressor()))
                # scaling the target variable through TransformedTargetRegressor
                # is not compatible with ONNX
            ]
        )

    def set_params(self, hyperparameters):
        # access the hyperparameters of the model within the pipeline within the
        # TransformedTargetRegressor
        self.regressor.named_steps["model"].regressor.set_params(**hyperparameters)

    def get_params(self, deep=True):
        # Get the hyperparameters of the model
        return self.regressor.named_steps["model"].regressor.get_params(deep=deep)


class LinearReg(BaseScikitLearnModel):
    """Linear Regression model"""

    def __init__(self):
        super().__init__(LinearRegression())

    def grid_search_hyperparameter(self):
        pass

    def optuna_hyperparameter_suggest(self, trial):
        pass
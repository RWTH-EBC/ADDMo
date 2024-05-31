import os
import joblib
import json
import sklearn
import pandas as pd
from abc import ABC
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from skl2onnx import to_onnx
from core.s3_model_tuning.models.abstract_model import AbstractMLModel
from core.s3_model_tuning.models.abstract_model import ModelMetadata
from sklearn.linear_model import LinearRegression
from extrapolation_detection.util.loading_saving import create_path_or_ask_to_override


class BaseScikitLearnModel(AbstractMLModel, ABC):
    """
    Base class for scikit-learn models.
    This class extends the AbstractMLModel, providing concrete implementations of
    common functionalities specific to scikit-learn models.

    Attributes:
        model (Pipeline): A scikit-learn Pipeline object containing the scaler and the provided model.
    """

    def __init__(self, regressor):
        """
        Create an instance of the scikit-learn model including a scaler
        """
        self.regressor = Pipeline(
            steps=[
                ("scaler", StandardScaler()),  # scale the features
                ("model", regressor)  # scaling the target variable through TransformedTargetRegressor
                # is not compatible with ONNX
            ]
        )

    def fit(self, x, y):
        self.x_fit = x
        self.y_fit = y
        self.regressor.fit(x.values, y)  # Train the model

    def predict(self, x):
        return self.regressor.predict(x)  # Make predictions

    def _define_metadata(self, directory, regressor_filename): # Todo: make this consistent, defining model
        """
        Define metadata.
        """
        self.metadata = ModelMetadata(
            addmo_class=type(self).__name__,
            addmo_commit_id=ModelMetadata.get_commit_id(),
            library=sklearn.__name__,
            library_model_type=type(self.regressor.named_steps['model']).__name__,
            library_version=sklearn.__version__,
            target_name=self.y_fit.name,
            features_ordered=list(self.x_fit.columns),
            preprocessing=['StandardScaler for all features'])

        # save metadata
        regressor_filename = os.path.splitext(regressor_filename)[0] #todo: for scikit and keras both delete, put to abstract
        metadata_path = os.path.join(directory, regressor_filename + '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata.dict(), f)

    def save_regressor(self, directory, filename=None, file_type='joblib'):
        """"
        Save regressor as .joblib or .onnx file
        """

        if filename is None:
            filename = type(self).__name__

        full_filename = f"{filename}.{file_type}"
        path = create_path_or_ask_to_override(full_filename, directory)

        if file_type == 'joblib':
            joblib.dump(self.regressor, path)
            self._define_metadata(directory, filename) # todo: not in if

        elif file_type == 'onnx':
            onnx_model = to_onnx(self.regressor, self.x_fit.values)
            self._define_metadata(directory, filename)
            with open(path, "wb") as f:
                f.write(onnx_model.SerializeToString())

        print(f"Model saved to {path}.")

    def load_regressor(self, regressor):
        self.regressor = regressor

    def to_scikit_learn(self, x=None):
        return self.regressor

    def set_params(self, hyperparameters):
        """
        access the hyperparameters of the model within the pipeline within the TransformedTargetRegressor
        """
        self.regressor.named_steps["model"].set_params(**hyperparameters)

    def get_params(self, deep=True):
        """
        Get the hyperparameters of the model
        """
        return self.regressor.named_steps["model"].get_params(deep=deep)


class ScikitMLP(BaseScikitLearnModel):
    """Scikit-learn MLPRegressor model."""

    def __init__(self):
        super().__init__(MLPRegressor())

    def optuna_hyperparameter_suggest(self, trial):
        hyperparameters = {}

        # Suggest hyperparameters
        n_layers = trial.suggest_int("n_layers", 1, 2)
        hidden_layer_sizes = tuple(
            trial.suggest_int(f"n_units_l{i}", 1, 1000) for i in range(n_layers)
        )

        # Dynamic hidden layer sizes based on the number of layers
        hyperparameters["hidden_layer_sizes"] = hidden_layer_sizes

        # Other hyperparameters
        hyperparameters["activation"] = "relu"
        hyperparameters["max_iter"] = 5000

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

    def default_hyperparameter(self):
        return MLPRegressor().get_params()


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

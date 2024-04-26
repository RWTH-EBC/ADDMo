import os
import skl2onnx
import joblib
import json
import sklearn
import subprocess
import pandas as pd
from abc import ABC
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from core.s3_model_tuning.models.metadata.modelmetadata import ModelMetadata
from skl2onnx import to_onnx
from core.s3_model_tuning.models.abstract_model import AbstractMLModel
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

        if not isinstance(x, pd.DataFrame):
            raise TypeError('x should be a pandas dataframe and not of type', type(x))
        if not isinstance(y, (pd.Series, pd.DataFrame)):
            raise TypeError('y should be a pandas dataframe/series and not of type', type(y))
        self.x = x  # Save the training data to be used later for ONNX conversion
        self.y = y  # Save the target column to get target name for metadata
        self.regressor.fit(x, y)  # Train the model

    def predict(self, x):
        return self.regressor.predict(x)  # Make predictions

    def addmo_class(self):
        return type(self).__name__

    def save_metadata(self, directory, filename):
        # Todo: metadata should be saved at same directory as the model, having the same file name with suffix <modelname_metadata>

        self.metadata = ModelMetadata(  # Todo:
            addmo_class=self.addmo_class(),
            addmo_commit_id=get_commit_id(),
            library=sklearn.__name__,  # dynamic: if we add keras class later
            library_model_type=type(self.regressor.named_steps['model']).__name__,
            library_version=sklearn.__version__,
            target_name=self.y.name,
            features_ordered=list(self.x.columns),
            preprocessing=['StandardScaler for all features'],
            instructions='Pass a single or multiple observations with features in the order listed above.')

        filename = os.path.splitext(filename)[0]
        if filename == self.addmo_class():
            metadata_path = os.path.join(directory, filename + '_metadata.json')
        else:
            metadata_path = os.path.join(directory, filename + '_metadata.json')  # + self.addmo_class()

        with open(metadata_path, 'w') as f:
            json.dump(self.metadata.dict(), f)
            # print('Metadata saved to', metadata_path)

    def save_regressor(self, directory, filename=None, file_type='joblib'):  # Todo: change accordingly
        # Todo: if filename none use the name of the model class

        if filename is None:
            filename = self.addmo_class() + '.' + file_type

        if filename.endswith('.joblib'):
            joblib.dump(self.regressor, os.path.join(directory, filename))
            self.save_metadata(directory, filename)
            print('Model saved to', os.path.join(directory, filename))

        elif filename.endswith('.onnx'):
            onx = to_onnx(self.regressor, self.x[:1])
            with open(os.path.join(directory, filename), "wb") as f:
                f.write(onx.SerializeToString())
                print('Model saved to',
                      os.path.join(directory, filename))  # Todo: print: model saved to {path+filename+type}

    # def get_filename(self, filename):
    # return

    def load_regressor(self, regressor):
        # Implement model loading
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


class MLP(BaseScikitLearnModel):
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


class MLP_TargetTransformed(MLP):
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
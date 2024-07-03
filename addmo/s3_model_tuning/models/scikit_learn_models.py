import joblib
import sklearn
from abc import ABC
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.compose import TransformedTargetRegressor
from skl2onnx import to_onnx
from addmo.s3_model_tuning.models.abstract_model import AbstractMLModel
from addmo.s3_model_tuning.models.abstract_model import ModelMetadata
from sklearn.linear_model import LinearRegression


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
        """
        Train the model.
        """
        self.x_fit = x
        self.y_fit = y
        self.regressor.fit(x.values, y)

    def predict(self, x):
        """
        Make predictions.
        """
        return self.regressor.predict(x.values)

    def _define_metadata(self):
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

    @property
    def default_file_type(self):
        """
        Set filetype for saving trained model.
        """
        return 'joblib'

    def _save_regressor(self, path, file_type):
        """"
        Save regressor as .joblib or .onnx file
        """

        if file_type == 'joblib':
            joblib.dump(self.regressor, path)

        elif file_type == 'onnx':
            onnx_model = to_onnx(self.regressor, self.x_fit.values)
            with open(path, "wb") as f:
                f.write(onnx_model.SerializeToString())

        print(f"Model saved to {path}.")

    def load_regressor(self, regressor):
        """""
        Load trained model for serialisation.
        """
        self.regressor = regressor

    def to_scikit_learn(self, x=None):
        return self.regressor

    def set_params(self, hyperparameters):
        """
        Access the hyperparameters of the model within the pipeline within the TransformedTargetRegressor
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
        self.set_params(self.default_hyperparameter())

    def optuna_hyperparameter_suggest(self, trial):
        hyperparameters = {}

        # Suggest hyperparameters
        n_layers = trial.suggest_int("n_layers", 1, 2)
        hidden_layer_sizes = tuple(
            trial.suggest_int(f"n_units_l{i}", 1, 1000) for i in range(n_layers)
        )

        # Dynamic hidden layer sizes based on the number of layers
        hyperparameters["hidden_layer_sizes"] = hidden_layer_sizes

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
        """"
        Return default hyperparameters.
        """
        hyperparameter = MLPRegressor().get_params()
        hyperparameter["max_iter"] = 5000
        hyperparameter["early_stopping"] = True
        return hyperparameter

class ScikitMLP_TargetTransformed(ScikitMLP):
    def __init__(self):
        """
        Create an instance of the scikit-learn model including a scaler.
        """
        self.regressor = Pipeline(
            steps=[
                ("scaler", StandardScaler()),  # scale the features
                ("model", TransformedTargetRegressor(regressor=MLPRegressor()))
                # scaling the target variable through TransformedTargetRegressor
                # is not compatible with ONNX
            ]
        )
        self.set_params(self.default_hyperparameter())

    def set_params(self, hyperparameters):
        """
        Access the hyperparameters of the model within the pipeline within the TransformedTargetRegressor.
        """
        self.regressor.named_steps["model"].regressor.set_params(**hyperparameters)

    def get_params(self, deep=True):
        """
        Get the hyperparameters of the model.
        """
        return self.regressor.named_steps["model"].regressor.get_params(deep=deep)

class ScikitLinearReg(BaseScikitLearnModel):
    """Linear Regression model"""

    def __init__(self):
        super().__init__(LinearRegression())

    def grid_search_hyperparameter(self):
        pass

    def optuna_hyperparameter_suggest(self, trial):
        pass

    def default_hyperparameter(self):
        """"
        Return default hyperparameters.
        """
        return LinearRegression().get_params()

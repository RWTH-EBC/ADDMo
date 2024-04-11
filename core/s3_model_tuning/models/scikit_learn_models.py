from abc import ABC

from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
import onnx
import skl2onnx

from core.s3_model_tuning.models.abstract_model import AbstractMLModel


class BaseScikitLearnModel(AbstractMLModel, ABC):
    """
    Base class for scikit-learn models.
    This class extends the AbstractMLModel, providing concrete implementations of
    common functionalities specific to scikit-learn models.

    Attributes:
        model (Pipeline): A scikit-learn Pipeline object containing the scaler and the provided model.
    """

    def __init__(self, model):
        # Create an instance of the scikit-learn model including a scaler
        self.model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),  # scale the features
                ("model", model) # scaling the target variable through TransformedTargetRegressor
                # is not compatible with ONNX
            ]
        )

    def fit(self, x, y):
        self.x = x  # Save the training data to be used later for ONNX conversion
        self.model.fit(x, y)  # Train the model

    def predict(self, x):
        return self.model.predict(x)  # Make predictions

    def save_model(self, abs_path):
        # Convert the entire pipeline to joblib
        onnx_model = skl2onnx.to_onnx(self.model, self.x[:1])
        onnx.save_model(onnx_model, abs_path)
        #create metadata file

    def load_model(self, abs_path):
        # Implement model loading
        self.model = load(abs_path)
        pass

    def to_scikit_learn(self):
        return self.model

    def set_params(self, hyperparameters):
        # access the hyperparameters of the model within the pipeline within the
        # TransformedTargetRegressor
        self.model.named_steps["model"].set_params(**hyperparameters)

    def get_params(self, deep=True):
        # Get the hyperparameters of the model
        return self.model.named_steps["model"].get_params(deep=deep)

    def default_hyperparameter(self):
        return self.model.get_params()


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
        self.model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),  # scale the features
                ("model", TransformedTargetRegressor(regressor=MLPRegressor())) # scaling the target variable through TransformedTargetRegressor
                # is not compatible with ONNX
            ]
        )

    def set_params(self, hyperparameters):
        # access the hyperparameters of the model within the pipeline within the
        # TransformedTargetRegressor
        self.model.named_steps["model"].regressor.set_params(**hyperparameters)

    def get_params(self, deep=True):
        # Get the hyperparameters of the model
        return self.model.named_steps["model"].regressor.get_params(deep=deep)

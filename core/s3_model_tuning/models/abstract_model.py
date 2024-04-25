import onnxruntime as rt
import numpy
from abc import ABC, abstractmethod


class AbstractMLModel(ABC):
    """
    Abstract base class for machine learning models.

    This class provides an interface for all machine learning models, potentially including
    a scaler.

    Attributes:
        model: An instance of the machine learning model, usually including the scaler.
    """

    @abstractmethod
    def __init__(self):
        """Initializes the machine learning model."""
        self.model = None

    @abstractmethod
    def fit(self, x, y):
        """
        Train the model on the provided data.

        Args:
            x: Features used for training.
            y: Target values used for training.
        """
        pass

    @abstractmethod
    def predict(self, x):
        """
        Make predictions on the given input data.

        Args:
            x: Input data for making predictions.

        Returns:
            Predicted values, scaled back to the original scale if applicable.
        """
        pass

    @abstractmethod
    def save_model(self, path):
        """
        Save the trained model including scaler to a file.
        This is done using the ONNX format.

        Args:
            path: File path where the model will be saved.
        """
        pass

    def load_model(self, model_instance): #Todo: delete this method possibly
        """
        Load a model including scaler.

        Args:
            model_instance: model that is loaded.
        """
        self.model = model_instance

    @abstractmethod
    def to_scikit_learn(self):
        """
        Convert the model including scaler to a scikit-learn compatible model.
        E.g. a scikit-learn pipeline.

        Most ML frameworks provide a converter to adapt models for scikit-learn specific tasks.

        Returns:
            A scikit-learn compatible version of the model including scaler.
        """
        pass

    @abstractmethod
    def set_params(self, **params):
        """
        Set the hyperparameters of the ML model.

        Args:
            **params: Variable length keyword arguments for hyperparameters.
        """
        pass

    @abstractmethod
    def get_params(self):
        """
        Get the hyperparameters of the ML model.

        Returns:
            A dictionary of the current hyperparameters.
        """
        pass

    @abstractmethod
    def optuna_hyperparameter_suggest(self, trial):
        """
        Suggest hyperparameters using Optuna for hyperparameter optimization.

        Args:
            trial: An Optuna trial object used to suggest hyperparameters.

        Returns:
            A dictionary of hyperparameters with Optuna distributions.
        """
        pass

    @abstractmethod
    def grid_search_hyperparameter(self):
        """
        Define the hyperparameters for grid search.

        Returns:
            A dictionary representing a hyperparameter grid for grid search.
        """
        pass

    @abstractmethod
    def default_hyperparameter(self):
        """
        Define the default hyperparameters of the model.

        Returns:
            A dictionary with a default set of hyperparameters.
        """
        pass


class PredictorOnnx(AbstractMLModel, ABC):
    """overwrites predict and load function for onnx format """

    def __init__(self):
        super().__init__()
        self.labels = None
        self.inputs = None
        self.session = None

    def load_model(self, path):
        self.session = rt.InferenceSession(path, providers=['CPUExecutionProvider'])
        self.inputs = self.session.get_inputs()[0].name
        self.labels = self.session.get_outputs()[0].name

    def predict(self, x):
        return self.session.run([self.labels], {self.inputs: x.astype(numpy.double)})[0]

    def default_hyperparameter(self):
        # Implement default hyperparameters
        pass

    def fit(self, x, y):
        # Implement model fitting
        pass

    def get_params(self):
        # Implement getting model hyperparameters
        pass

    def grid_search_hyperparameter(self):
        # Implement defining hyperparameters for grid search
        pass

    def optuna_hyperparameter_suggest(self, trial):
        # Implement suggesting hyperparameters using Optuna
        pass

    def save_model(self, path):
        # Implement saving the model
        pass

    def set_params(self, **params):
        # Implement setting model hyperparameters
        pass

    def to_scikit_learn(self):
        # Implement converting the model to scikit-learn compatible format
        pass

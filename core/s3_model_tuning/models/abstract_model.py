import warnings
import onnxruntime as rt
import numpy as np
import subprocess
from abc import ABC, abstractmethod

import pandas as pd
from pydantic import BaseModel, Field


class AbstractMLModel(ABC):
    """
    Abstract base class for machine learning models.

    This class provides an interface for all machine learning models, potentially including
    a scaler.

    Attributes:
        regressor: An instance of the machine learning model, usually including the scaler.
    """

    @abstractmethod
    def __init__(self):
        """Initializes the machine learning model."""
        self.regressor = None
        self.x_fit: pd.DataFrame = None
        self.y_fit: pd.DataFrame = None

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
    def save_regressor(self, path):
        """
        Save the trained model including scaler to a file.
        This is done using the ONNX format.

        Args:
            path: File path where the model will be saved.
        """
        pass

    def load_regressor(self, model_instance):
        """
        Load a model including scaler.

        Args:
            model_instance: model that is loaded.
        """
        self.regressor = model_instance

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
    """overwrites predict and load function for onnx format"""

    def __init__(self):
        super().__init__()
        self.labels = None
        self.inputs = None
        self.model = None

    def load_regressor(self, path):
        self.model = rt.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.inputs = self.model.get_inputs()[0].name
        self.labels = self.model.get_outputs()[0].name

    def predict(self, x):
        x_ONNX= x.values  # Converts dataframe to numpy array
        return self.model.run([self.labels], {self.inputs: x_ONNX.astype(np.float32)})[0]

    def default_hyperparameter(self):
        warnings.warn(f"This function is not implemented for ONNX models")

    def fit(self, x, y):
        warnings.warn(f"This function is not implemented for ONNX models")

    def get_params(self):
        warnings.warn(f"This function is not implemented for ONNX models")

    def grid_search_hyperparameter(self):
        warnings.warn(f"This function is not implemented for ONNX models")

    def optuna_hyperparameter_suggest(self, trial):
        warnings.warn(f"This function is not implemented for ONNX models")

    def save_regressor(self, path):
        warnings.warn(f"This function is not implemented for ONNX models")

    def set_params(self, **params):
        warnings.warn(f"This function is not implemented for ONNX models")

    def to_scikit_learn(self):
        warnings.warn(f"This function is not implemented for ONNX models")


class ModelMetadata(BaseModel):
    """ModelMetadata class represents metadata associated with the trained machine
    learning model when saved in joblib format."""

    addmo_class: str = Field(
        description="ADDMo model class type, from which the regressor was saved."
    )
    addmo_commit_id: str = Field(
        description="Current commit id when the model is saved."
    )
    library: str = Field(description="ML library origin of the regressor")
    library_model_type: str = Field(description="Type of regressor within library")
    library_version: str = Field(description="library version used")
    target_name: str = Field(description="Name of the target variable")
    features_ordered: list = Field(description="Name and order of features")
    preprocessing: list = Field(
        description="Preprocessing steps applied to the features."
    )
    instructions: str = Field(
        "Pass a single or multiple observations with features in the order listed above",
        description="Instructions for passing input data for making predictions.",
    )

    def get_commit_id():
        """Get the commit id for metadata when model is saved. """

        try:
            commit_id = subprocess.check_output(["git", "describe", "--always"]).strip().decode()
        except subprocess.CalledProcessError:
            commit_id = 'Unknown'
        return commit_id

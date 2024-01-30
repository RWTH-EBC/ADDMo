from abc import ABC, abstractmethod


class AbstractMLModel(ABC):
    """
    Abstract base class for machine learning models.
    Should include, if required, a scaler.
    """

    @abstractmethod
    def __init__(self):
        self.model = None

    @abstractmethod
    def fit(self, x, y):
        """
        Train the model on the provided data.
        """
        pass

    @abstractmethod
    def predict(self, x):
        """
        Make predictions on the given input data.
        The prediction should be scaled back to the original scale.
        """
        pass

    @abstractmethod
    def save_model(self, path):
        """
        Save the trained model to a file.
        """
        pass

    @abstractmethod
    def load_model(self, path):
        """
        Load a model from a file.
        """
        pass

    @abstractmethod
    def to_scikit_learn(self):
        """
        Convert the model to a scikit-learn model for several scikit_learn specific tasks.
        Most ML frameworks provide such converter.
        """
        pass

    @abstractmethod
    def set_params(self, **params):
        """
        Set the hyperparameters of the ML model.
        """
        pass

    @abstractmethod
    def get_params(self):
        """
        Get the hyperparameters of the ML model.
        """
        pass

    @abstractmethod
    def optuna_hyperparameter_suggest(self, trial):
        """
        Suggest hyperparameters for Optuna´s hyperparameter optimization.
        Returns a dictionary of hyperparameters with optuna distributions.
        """
        pass

    @abstractmethod
    def grid_search_hyperparameter(self):
        """
        Define the hyperparameters for grid search.
        Returns a dictionary of a hyperparameter grid.
        """
        pass

    @abstractmethod
    def default_hyperparameter(self):
        """
        Define the default hyperparameters of the model.
        Returns a dictionary with one set of hyperparameters.
        """
        pass

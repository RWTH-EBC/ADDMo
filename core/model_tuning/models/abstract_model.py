from abc import ABC, abstractmethod

class AbstractMLModel(ABC):
    """
    Abstract base class for machine learning models.
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
    def infer(self, x):
        """
        Make predictions on the given input data.
        """
        pass

    def optuna_hyperparameter_suggest(self, trial):
        """
        Suggest hyperparameters for Optuna´s hyperparameter optimization.
        Returns a dictionary of hyperparameters with optuna distributions.
        """
        pass

    def grid_search_hyperparameter(self):
        """
        Define the hyperparameters for grid search.
        Returns a dictionary of a hyperparameter grid.
        """
        pass

    def default_hyperparameter(self):
        """
        Define the default hyperparameters of the model.
        Returns a dictionary with one set of hyperparameters.
        """
        pass

    @abstractmethod
    def set_hyperparameters(self, hyperparameters:dict):
        """
        Set the hyperparameters of the model.
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

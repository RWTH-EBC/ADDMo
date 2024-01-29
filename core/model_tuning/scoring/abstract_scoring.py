from abc import ABC, abstractmethod
from core.model_tuning.models.abstract_model import AbstractMLModel



class AbstractScoring(ABC):

    @abstractmethod
    @staticmethod
    def score(model: AbstractMLModel, x, y):
        """ Returns a positive float value. The higher the better.
        x and y include train and evaluation period."""
        pass

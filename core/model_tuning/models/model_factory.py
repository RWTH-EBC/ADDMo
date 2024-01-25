from core.model_tuning.models.scikit_learn_models import MLP
from core.model_tuning.models.abstract_model import AbstractMLModel

class ModelFactory:
    """
    Creates and returns an instance of the specified machine learning model.
    """

    @staticmethod
    def model_factory(model_type: str) -> AbstractMLModel:
        if model_type == 'MLP':
            return MLP()
        elif model_type == 'test':
            pass
        # Add more conditions for other models
        else:
            raise ValueError("Unknown model type")

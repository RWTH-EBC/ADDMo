import inspect

from core.s3_model_tuning.models import scikit_learn_models
from core.s3_model_tuning.models.abstract_model import AbstractMLModel


class ModelFactory:
    """
    Creates and returns an instance of the specified machine learning model.
    """

    @staticmethod
    def model_factory(model_type: str) -> AbstractMLModel:
        """Get the model instance dynamically."""

        # If model is based on scikit-learn
        if hasattr(scikit_learn_models, model_type):
            custom_model_class = getattr(scikit_learn_models, model_type)
            return custom_model_class()

        # You may add something like:
        # # If model is based on e.g. Keras
        # elif hasattr(keras_models, model_type):
        #     custom_model_class = getattr(keras_models, model_type)
        #     return custom_model_class(**kwargs)

        # If model is not found
        else:
            # Get the names of all custom models for error message
            custom_model_names = [
                name
                for name, obj in inspect.getmembers(scikit_learn_models)
                if inspect.isclass(obj)
                and issubclass(obj, AbstractMLModel)
                and not inspect.isabstract(obj)
            ]

            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Available custom models are: {', '.join(custom_model_names)}. "
            )

    def load_serialized_model(self, path):
        '''#TODO'''
        # get addmo class name from metadata

        # get addmo model class from factory

        # load serialized model, e.g. scikit, to addmo model class

        return model (addmo model)

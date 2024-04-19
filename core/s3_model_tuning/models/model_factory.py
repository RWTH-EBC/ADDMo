import inspect
import sys
from core.s3_model_tuning.models import scikit_learn_models
from core.s3_model_tuning.models.abstract_model import AbstractMLModel
import json

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

        #dynamically create instances of apt addmo class based on the info extracted from the metadata and then load it

        '''#TODO'''
        # get addmo class name from metadata
        if not path.endswith('.joblib'):
            raise ValueError(" '.joblib' path expected")

        metadata_path = path + '.json'
        with open(metadata_path) as f:
            metadata = json.load(f)

        addmo_class = metadata.get('addmo_class')

        # get addmo model class from factory
        addmo_model_class=  ModelFactory.model_factory(addmo_class)

        # load serialized model, e.g. scikit, to addmo model class
        addmo_model_class.load_model(path)
        print('loaded addmo class model')

        return addmo_model_class


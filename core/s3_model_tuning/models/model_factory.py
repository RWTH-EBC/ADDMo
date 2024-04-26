import inspect
import sys
import json
import joblib
import os
from core.s3_model_tuning.models import scikit_learn_models
from core.s3_model_tuning.models.abstract_model import AbstractMLModel
from core.s3_model_tuning.models.abstract_model import PredictorOnnx
from core.util.definitions import root_dir


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

    @staticmethod
    def load_model(abs_path: str) -> AbstractMLModel:
        '''Load the model from the specified path and return the model instance.'''

        # Load regressor from joblib file to addmo model class
        if abs_path.endswith('.joblib'):
            metadata_path = f"{abs_path}_metadata.json"

            if os.path.exists(metadata_path):
                with open(metadata_path) as f:
                    metadata = json.load(f)
            else:
                raise FileNotFoundError(
                    f'The metadata file {metadata_path} does not exist. Try saving the model before loading it or specify the path where model is saved ')

            addmo_class_name = metadata.get('addmo_class')
            addmo_class = ModelFactory.model_factory(addmo_class_name)
            regressor = joblib.load(abs_path)
            addmo_class.load_regressor(regressor)

        # Load the regressor from onnx file to PredictorOnnx class
        elif abs_path.endswith('.onnx'):
            addmo_class = PredictorOnnx()
            addmo_class.load_regressor(abs_path)

        else:
            raise ValueError(" '.joblib' or '.onnx' path expected")

        return addmo_class

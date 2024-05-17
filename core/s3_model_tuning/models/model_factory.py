import inspect
import sys
import json
import joblib
import os
from core.s3_model_tuning.models import scikit_learn_models
from core.s3_model_tuning.models import keras_model
from core.s3_model_tuning.models.abstract_model import AbstractMLModel
from core.s3_model_tuning.models.abstract_model import PredictorOnnx
from tensorflow import keras
from tensorflow.keras.models import load_model

class ModelFactory:
    """
    Creates and returns an instance of the specified machine learning model.
    """

    @staticmethod
    def model_factory(model_type: str, **kwargs) -> AbstractMLModel:
        """Get the model instance dynamically."""

        # If model is based on scikit-learn
        if hasattr(scikit_learn_models, model_type):
            if kwargs:
                raise ValueError("No keyword arguments allowed for scikit-learn models.")
            custom_model_class = getattr(scikit_learn_models, model_type)
            return custom_model_class()

        #  If model is based on e.g. Keras
        elif hasattr(keras_model, model_type):
             custom_model_class = getattr(keras_model, model_type)
             return custom_model_class(**kwargs)

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

    def _load_metadata(abs_path: str):
        """Read metatdata file when model is loaded."""

        filename = os.path.splitext(abs_path)[0]
        metadata_path = f"{filename}_metadata.json"

        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                metadata = json.load(f)
            return metadata
        else:
            raise FileNotFoundError (f'The metadata file {metadata_path} does not exist. Try saving the model before loading it or specify the path where the model is saved.')


    @staticmethod
    def load_model(abs_path: str) -> AbstractMLModel:
        '''Load the model from the specified path and return the model instance.'''

        # Load regressor from joblib file to addmo model class
        if abs_path.endswith('.joblib'):
            metadata = ModelFactory._load_metadata(abs_path)
            addmo_class_name = metadata.get('addmo_class')
            addmo_class = ModelFactory.model_factory(addmo_class_name)
            regressor = joblib.load(abs_path)
            addmo_class.load_regressor(regressor)

        # Load the regressor from onnx file to PredictorOnnx class
        elif abs_path.endswith('.onnx'):
            addmo_class = PredictorOnnx()
            addmo_class.load_regressor(abs_path)

        # Load the regressor from keras file to addmo model class
        elif abs_path.endswith('.h5'):
            metadata = ModelFactory._load_metadata(abs_path)
            addmo_class_name = metadata.get('addmo_class')
            addmo_class = ModelFactory.model_factory(addmo_class_name)
            regressor= keras.models.load_model(abs_path)
            addmo_class.load_regressor(regressor)

        else:
            raise ValueError(" '.joblib', '.onnx' or '.h5' path expected")

        return addmo_class

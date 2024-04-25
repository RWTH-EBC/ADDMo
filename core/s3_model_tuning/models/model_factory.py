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

    def load_model(self, path):
 #Todo:
        # this is not dynamic! Here should be something like "if directory of path
        # contains metadata load it, else print a raise an descriptive error", also this whole meta data
        # thing is only required for joblib, not for ONNX, so put it into the respective if clause.

        if path.endswith('.joblib'):

            metadata_path = os.path.join(root_dir(), 'core', 's3_model_tuning', 'models',
                                         'metadata', path + '.json')
            if Path(metadata_path).is_file():
                with open(metadata_path) as f:
                    metadata = json.load(f)
            else:
                raise FileNotFoundError(f'The metadata file {metadata_path} does not exist. ')
            addmo_class = metadata.get('addmo_class')
            addmo_model_class = ModelFactory.model_factory(addmo_class)
            model = joblib.load(path)
            addmo_model_class.load_regressor(model)

        elif path.endswith('.onnx'):
            addmo_model_class = PredictorOnnx()
            addmo_model_class.load_regressor(path)

        else:
            raise ValueError(" '.joblib' or '.onnx' path expected")

        return (addmo_model_class)

import os
import json
import keras
import tensorflow as tf
import onnx
import pandas as pd
import h5py
from abc import ABC
from packaging import version
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Normalization, Activation
from scikeras.wrappers import KerasRegressor
from keras.losses import MeanSquaredError
from core.s3_model_tuning.models.abstract_model import AbstractMLModel
from core.s3_model_tuning.models.abstract_model import ModelMetadata
from core.util.load_save_utils import create_path_or_ask_to_override


class BaseKerasModel(AbstractMLModel, ABC):
    """
    Base class for Keras models.
    """

    def _define_metadata(self, directory, regressor_filename):
        """
        Define metadata.
        """
        self.metadata = ModelMetadata(
            addmo_class=type(self).__name__,
            addmo_commit_id=ModelMetadata.get_commit_id(),
            library=keras.__name__,
            library_model_type='Sequential',
            library_version=keras.__version__,
            target_name=self.y_fit.name,
            features_ordered=list(self.x_fit.columns),
            preprocessing=['Scaling as layer of the ANN.'])

        # Save Metadata.
        metadata_path = os.path.join(directory, regressor_filename + '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata.dict(), f)

    def save_regressor(self, directory, filename=None, file_type='h5'):
        """
        Save regressor as a .h5 or .keras file.
        """
        if filename is None:
            filename = type(self).__name__
        self._define_metadata(directory, filename)
        full_filename = f"{filename}.{file_type}"

        path = create_path_or_ask_to_override(full_filename, directory)

        if file_type in ['h5', 'keras']:
            self.regressor.model_.save(path, overwrite=True)
        elif file_type == "onnx":
            # catch exceptions
            if version.parse(keras.__version__).major != 2:  # Checking version compatibility
                raise ImportError("ONNX is only supported with Keras version 2")
            try:
                import tf2onnx
            except ImportError:
                raise ImportError("tf2onnx is required to save the model in ONNX format")

            # actually save onnx
            spec = (tf.TensorSpec((None,) + self.regressor.model_.input_shape[1:], tf.float32, name="input"),)
            onnx_model, _ = tf2onnx.convert.from_keras(self.regressor.model_, input_signature=spec, opset=13)
            onnx.save(onnx_model, path)
        print(f"Model saved to {path}")

    def load_regressor(self, regressor):
        """""
        Load trained model for serialisation.
        """
        self.regressor = regressor


class SciKerasSequential(BaseKerasModel):
    """"" SciKeras Sequential model. """

    def __init__(self):
        self.regressor = KerasRegressor()  # SciKeras Regressor
        self.hyperparameters = self.default_hyperparameter()

    def fit(self, x, y):
        """""
        Build, compile and fit the sequential model.
        """
        self.x_fit = x
        self.y_fit = y
        input_shape = (len(x.columns),)
        sequential_regressor = self._build_regressor(input_shape)
        # Normalisation of first layer (input data).
        sequential_regressor.layers[0].adapt(x.to_numpy())  # Normalisation initialisation works only on np arrays
        self.regressor = KerasRegressor(model=sequential_regressor,
                                        loss=self.hyperparameters['loss'],
                                        epochs=self.hyperparameters['max_iter'],
                                        verbose=0)
        self.regressor.fit(x, y)

    def predict(self, x):
        return self.regressor.predict(x)

    def get_params(self, deep=True):
        """
        Get the hyperparameters of the model.
        """
        return self.regressor.get_params()

    def set_params(self, hyperparameters):
        """""
        Update the hyperparameters in internal storage, which is accessed while building the
        regressor. Not done here, because compilation requires the input_shape to be available.
        """
        self.hyperparameters = hyperparameters

    def _build_regressor_architecture(self, input_shape):
        """
        Builds a sequential model.
        """
        sequential_regressor = Sequential()
        normalizer = Normalization(axis=-1)  # Preprocessing layer
        sequential_regressor.add(Input(shape=input_shape))
        sequential_regressor.add(normalizer)
        # Adding hidden layers based on hyperparameters
        for units in self.hyperparameters['hidden_layer_sizes']:
            sequential_regressor.add(Dense(units=units, activation='relu'))

        sequential_regressor.add(Dense(1, activation='linear'))  # Output shape = 1 for continuous variable
        return sequential_regressor

    def _build_regressor(self, input_shape):
        """""
        Returns a compiled sequential model.
        """
        sequential_regressor = self._build_regressor_architecture(input_shape)
        sequential_regressor.compile(loss=self.hyperparameters['loss'])
        return sequential_regressor

    def to_scikit_learn(self, x):
        """""
        Convert Keras Model to Scikeras Regressor for tuning.
        """
        input_shape = (len(x.columns),)
        # proper compilation of the model is necessary for the conversion
        regressor_scikit = KerasRegressor(model=self._build_regressor_architecture(input_shape),
                                          loss=self.hyperparameters['loss'],
                                          epochs=self.hyperparameters['max_iter'],
                                          verbose=0)
        return regressor_scikit

    def default_hyperparameter(self):
        """"
        Return default hyperparameters.
        """
        hyperparameters = self.regressor.get_params()
        # Define default loss if not present
        if hyperparameters['loss'] is None:
            hyperparameters['loss'] = MeanSquaredError()
        hyperparameters['hidden_layer_sizes'] = (64,)  # Set default hidden layer size
        hyperparameters['max_iter'] = hyperparameters['epochs']  # Keras uses epochs as max_iterations
        return hyperparameters

    def optuna_hyperparameter_suggest(self, trial):

        n_layers = trial.suggest_int("n_layers", 1, 2)
        hidden_layer_sizes = tuple(trial.suggest_int(f"n_units_l{i}", 1, 10) for i in range(1, n_layers + 1, 1))
        hyperparameters = {
            "hidden_layer_sizes": hidden_layer_sizes,
            "loss": MeanSquaredError(),
            "max_iter": 5
        }
        return hyperparameters

    def grid_search_hyperparameter(self):

        hyperparameter_grid = {
            "hidden_layer_sizes": [(64,), (128, 64), (256, 128, 64)],
            "loss": [MeanSquaredError()],
            "max_iter": [5000]
        }
        return hyperparameter_grid

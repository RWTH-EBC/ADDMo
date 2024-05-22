import os
import json
import keras
import tensorflow as tf
import tf2onnx
import onnx
import pandas as pd
import h5py
from abc import ABC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Normalization, Dropout, Activation
from scikeras.wrappers import KerasRegressor
from keras.losses import MeanSquaredError
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Adagrad
from core.s3_model_tuning.models.abstract_model import AbstractMLModel
from core.s3_model_tuning.models.abstract_model import ModelMetadata


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
            target_name=self.y.name,
            features_ordered=list(self.x.columns),
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
        if file_type in ['h5', 'keras']:
            path = os.path.join(directory, f"{filename}.{file_type}")
            self.regressor.model_.save(path, overwrite=True)
        elif file_type == "onnx":
            path = os.path.join(directory, f"{filename}.{file_type}")
            onnx_model, _ = tf2onnx.convert.from_keras(self.regressor)
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
        self.x = x
        self.y = y
        input_shape = (len(x.columns),)
        sequential_regressor = self._build_regressor(input_shape)
        # Normalisation of first layer (input data).
        sequential_regressor.layers[0].adapt(x.to_numpy())  # Normalisation initialisation works only on np arrays
        self.regressor = KerasRegressor(model=sequential_regressor,
                                        optimizer=self.hyperparameters['optimizer'],
                                        loss=self.hyperparameters['loss'],
                                        verbose=0)
        self.regressor.fit(x, y)

    def predict(self, x):
        return self.regressor.predict(x)

    def get_params(self, deep=True):
        """
        Get the hyperparameters of the model.
        """
        return self.regressor.get_params()

    def _build_regressor_architecture(self, input_shape):
        """
        Builds a sequential model.
        """
        sequential_regressor = Sequential()
        normalizer = Normalization(axis=-1)  # Preprocessing layer
        sequential_regressor.add(Input(shape=input_shape))
        sequential_regressor.add(normalizer)
        sequential_regressor.add(Dense(units=64, activation=self.hyperparameters.get('activation')))
        sequential_regressor.add(Dropout(0.5))
        sequential_regressor.add(Dense(1, activation='linear'))  # Output shape = 1 for continuous variable
        return sequential_regressor

    def _build_regressor(self, input_shape):
        """""
        Returns a compiled sequential model.
        """
        sequential_regressor = self._build_regressor_architecture(input_shape)
        sequential_regressor.compile(optimizer=self.hyperparameters['optimizer'], loss=self.hyperparameters['loss'])
        return sequential_regressor

    def to_scikit_learn(self, x):
        """""
        Convert Keras Model to Scikeras Regressor for tuning.
        """
        input_shape = (len(x.columns),)
        # proper compilation of the model is necessary for the conversion
        self.regressor = KerasRegressor(model=self._build_regressor_architecture(input_shape),
                                        optimizer=self.hyperparameters['optimizer'],
                                        loss=self.hyperparameters['loss'],
                                        verbose=0)

        return self.regressor

    def set_params(self, hyperparameters):
        """""
        Update the hyperparameters in internal storage, which is accessed while building the
        regressor. Not done here, because compilation requires the input_shape to be available.
        """
        self.hyperparameters = hyperparameters


    def default_hyperparameter(self):
        """"
        Return default hyperparameters.
        """
        hyperparameters = self.regressor.get_params()
        # Define default loss if not present
        if hyperparameters['loss'] is None:
            hyperparameters['loss'] = MeanSquaredError()

        return hyperparameters

    def optuna_hyperparameter_suggest(self, trial):
        hyperparameters = {
            "activation": trial.suggest_categorical("activation", ["relu", "sigmoid", "linear", "tanh"]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
            "optimizer": trial.suggest_categorical("optimizer", ["sgd", "adam", "rmsprop"]),
            "loss": MeanSquaredError()
        }
        return hyperparameters

    def grid_search_hyperparameter(self):
        hyperparameter_grid = {
            "activation": ["tanh", "relu", "softmax", "leaky_relu", "sigmoid", "exponential"],
            "optimizer": [SGD(), Adam(), RMSprop(), Adagrad()],
            "learning_rate": [0.0001, 0.001, 0.01],
            "loss": MeanSquaredError()
        }
        return hyperparameter_grid


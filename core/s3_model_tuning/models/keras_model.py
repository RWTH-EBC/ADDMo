import os
import json
from abc import ABC
import keras
import pandas as pd
import h5py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Normalization, Dropout, Activation
from scikeras.wrappers import KerasRegressor
from keras.losses import MeanSquaredError
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
        path = os.path.join(directory, f"{filename}.{file_type}")
        self._define_metadata(directory, filename)
        self.regressor.save(path, overwrite=True)
        print(f"Model saved to {path}")

    def load_regressor(self, regressor):
        """""
        Load trained model for serialisation.
        """
        self.regressor = regressor


class SciKerasSequential(BaseKerasModel):
    """"" SciKeras Sequential model. """

    def __init__(self):
        self.regressor = KerasRegressor()
        self.hyperparameters = self.default_hyperparameter()

    def fit(self, x, y):
        """""
        Build, compile and fit the sequential model.
        """
        self.x = x
        self.y = y
        self.input_shape = (len(x.columns),)
        self.regressor = self._build_regressor(self.hyperparameters, self.input_shape)
        # Normalisation of first layer (input data).
        self.regressor.layers[0].adapt(x.to_numpy())
        # Normalisation works on np arrays
        self.regressor.fit(x, y)

    def predict(self, x):
        return self.regressor.predict(x)

    def get_params(self, deep=True):
        """
        Get the hyperparameters of the model.
        """
        return self.regressor.get_params()

    def _build_regressor_architecture(self, hyperparameters, input_shape):
        """
        Builds a sequential model.
        """
        sequential_regressor = Sequential()
        normalizer = Normalization(axis=-1)  # Preprocessing layer
        sequential_regressor.add(Input(shape=input_shape))
        sequential_regressor.add(normalizer)
        sequential_regressor.add(Dense(units=64, activation=hyperparameters.get('activation')))
        sequential_regressor.add(Dropout(hyperparameters.get('dropout', 0.5)))
        sequential_regressor.add(Dense(1, activation='linear'))  # Output shape = 1 for continuous variable
        return sequential_regressor

    def _build_regressor(self, hyperparameters, input_shape):
        """""
        Returns a compiled sequential model.
        """
        sequential_regressor = self._build_regressor_architecture(hyperparameters, input_shape)
        sequential_regressor.compile(optimizer=hyperparameters['optimizer'], loss=hyperparameters['loss'])
        return sequential_regressor

    def _to_scikeras(self, hyperparameters):
        """
        Wrap the keras model to scikeras regressor for initialisation with updated hyperparameter.
        """
        self.regressor = KerasRegressor(
            model=lambda: self._build_regressor_architecture(hyperparameters),
            optimizer=hyperparameters['optimizer'],
            loss=hyperparameters['loss'],
            verbose=0)
        return self.regressor

    def to_scikit_learn(self, x):
        """""
        Convert Keras Model to Scikeras Regressor for tuning.
        """
        self.input_shape = (len(x.columns),)
        self.regressor = KerasRegressor(model=self._build_regressor_architecture
        (self.hyperparameters, self.input_shape),
                                        optimizer=self.hyperparameters['optimizer'],
                                        loss=self.hyperparameters['loss'],
                                        verbose=0)
        return self.regressor

    def set_params(self, hyperparameters):
        """""
        Update the hyperparameters and regressor.
        """
        self.hyperparameters = hyperparameters
        self.regressor = self._to_scikeras(hyperparameters)

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


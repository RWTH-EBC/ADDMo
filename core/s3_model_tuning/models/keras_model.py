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

    def save_features(self, x, y): #Todo: -> in abstract class!? and without the input_shape - Naming not descriptive
        # Save feature name and target name metadata from training data.
        self.feature_names = x.columns  # Save the feature names
        self.target_name = y.name  # Save the target name
        self.input_shape = (len(x.columns),)  # Save shape of training data for building regressor as a tuple

    def _save_metadata(self, directory, regressor_filename): # TOdo: rename only define metadata
        # Define Metadata.
        self.metadata = ModelMetadata(
            addmo_class=type(self).__name__,
            addmo_commit_id=ModelMetadata.get_commit_id(),
            library=keras.__name__,
            library_model_type='Sequential',
            library_version=keras.__version__,
            target_name=self.target_name,
            features_ordered=list(self.feature_names),
            preprocessing=['Scaling as layer of the ANN.'])

        # Save Metadata.
        regressor_filename = os.path.splitext(regressor_filename)[0]  # Remove file extension # Todo: necessary?
        metadata_path = os.path.join(directory, regressor_filename + '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata.dict(), f)

    def save_regressor(self, directory, filename=None, file_type='h5'):
        # Save model as a `.h5` file #TODO proper doc string! also onnx should be possible!
        if filename is None:
            filename = type(self).__name__
        path = os.path.join(directory, f"{filename}.{file_type}")
        self._save_metadata(directory, filename) # Todo: you tried this? filename has no extension, so why remove it in save metadata
        self.regressor.save(path, overwrite=True)
        print(f"Model saved to {path}")

    def load_regressor(self, regressor):
        # Load trained model for serialisation.
        self.regressor = regressor


class SciKerasSequential(BaseKerasModel):
    def __init__(self):
        self.regressor = KerasRegressor()

        self.hyperparameters = self.default_hyperparameter()
        self.hyperparameters['loss'] = MeanSquaredError()  # Default loss=none so update it here for compiling #Todo: why not in default_hyperparameter?

    def fit(self, x, y):
        self.x = x
        self.y = y #Todo: i feel like this is more general, and retrieve the respective info whereever needed

        self._build_regressor(self.hyperparameters, self.input_shape)
        # Normalisation of first layer (input data).
        self.regressor.layers[0].adapt(x.to_numpy())  # Normalisation works on np arrays
        self.regressor.fit(x, y)

    def predict(self, x):
        return self.regressor.predict(x)

    def get_params(self, deep=True):
        # Get the hyperparameters of the model. #Todo: in general, these should all be doc strings not comments if describing the function. I know i have not been consistent with this, but i will be in the future.
        return self.regressor.get_params()

    def _build_regressor_architecture(self, hyperparameters, input_shape): #Todo: sometimes you take hyperparameter from self and sometimes passed to function, clean up be consistent
        # Add layers to model.
        regressor = Sequential()
        normalizer = Normalization(axis=-1)  # Preprocessing layer
        regressor.add(Input(shape=input_shape))
        regressor.add(normalizer)
        regressor.add(Dense(units=64, activation=hyperparameters.get('activation', 'relu')))
        regressor.add(Dropout(hyperparameters.get('dropout', 0.5)))
        regressor.add(Dense(1, activation='linear'))  # Output shape = 1 for continuous variable
        return regressor

    def _compile_model(self, hyperparameters):
        self.regressor.compile(optimizer=hyperparameters['optimizer'], loss=hyperparameters['loss']) #Todo: this is not the self.regressor cause it the keras model not the scikeras! Consistency!

    def _build_regressor(self, hyperparameters, input_shape):
        self.regressor = self._build_regressor_architecture(self.hyperparameters, input_shape)
        self._compile_model(self.hyperparameters)

    def _to_scikeras(self): # should be inside build_regressor
        # Wrap the keras model to scikeras regressor for hyperparameter updating.
        self.regressor = KerasRegressor(
            model=lambda: self._build_regressor_architecture(self.hyperparameters),
            optimizer=self.hyperparameters['optimizer'],
            loss=self.hyperparameters['loss'],
            verbose=0)
        return self.regressor

    def to_scikit_learn(self, x):
        # Convert Keras Model to Scikeras Regressor.
        self.input_shape = (len(x.columns),)
        self.regressor = KerasRegressor(model=self._build_regressor_architecture
        (self.hyperparameters, self.input_shape),
                                        optimizer=self.hyperparameters['optimizer'],
                                        loss=self.hyperparameters['loss'],
                                        verbose=0)
        return self.regressor

    def set_params(self, hyperparameters):
        # Update the hyperparameters and regressor.
        self.hyperparameters = hyperparameters
        self.regressor = self._to_scikeras() # how will this work if not compiled? can you compile it afterwards? clean up with keras and scikeras

    def default_hyperparameter(self):
        params = self.regressor.get_params()
        if params "loss" = None -> define #Todo: see init, why not here?

        return params

    def optuna_hyperparameter_suggest(self, trial): #TODO: try to cover roughly the same hyperparameter options as for the scikit learn ANN
        hyperparameters = {
            "activation": trial.suggest_categorical("activation", ["relu", "sigmoid", "linear", "tanh"]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.5, step=0.1),
            "optimizer": trial.suggest_categorical("optimizer", ["sgd", "adam", "rmsprop"]),
            "loss": MeanSquaredError()
        }
        return hyperparameters

    def grid_search_hyperparameter(self):
        hyperparameter_grid = {
            "activation": ["tanh", "relu", "softmax", "leaky_relu", "sigmoid", "exponential"],
            "optimizer": [SGD(), Adam(), RMSprop(), Adagrad()],
            "epochs": [10, 50, 100, 150],
            "dropout_rate": [0.1, 0.2, 0.3, 0.4, 0.5],
            "learning_rate": [0.0001, 0.001, 0.01],
            "loss": MeanSquaredError()
        }
        return hyperparameter_grid

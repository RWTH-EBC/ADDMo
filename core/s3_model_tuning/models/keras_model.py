import os
import json
import optuna
from abc import ABC
import keras
import pandas as pd
from tensorflow.keras.models import Sequential
from scikeras.wrappers import KerasRegressor
from tensorflow.keras.layers import Dense, Dropout, Activation
from keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Adagrad
from core.s3_model_tuning.models.abstract_model import AbstractMLModel
from core.s3_model_tuning.models.abstract_model import ModelMetadata
from core.s3_model_tuning.models.abstract_model import get_commit_id


class BaseKerasModel(AbstractMLModel, ABC):
    """
    Base class for Keras models.
    This class extends the AbstractMLModel, providing concrete implementations of
    common functionalities specific to keras sequential model.

    Attributes:
        model: A keras model.
    """

    def __init__(self):
        # ask Martin: what should be the initialisation?
        self.regressor = None
        self.sklearn_regressor = None

    def _build_regressor(self):
        # Add layers to model : similar to MLP
        regressor = Sequential()
        regressor.add(Dense(units=64, input_shape=(len(self.feature_names),)))
        regressor.add(Activation('relu'))
        regressor.add(Dropout(0.5))
        regressor.add(Dense(1, activation='linear'))   # Output shape = 1 for continuous variable
        return regressor

    def compile_model(self, learning_rate=0.0001, epochs=10):
        self.learning_rate = learning_rate  # ask martin: can set params directly?
        self.epochs = epochs  # Save hyperparameters for get_hyperparameters()
        self.optimizer = SGD(learning_rate=learning_rate)  # Create an instance of SGD optimizer
        self.batch_size= 128
        self.regressor.compile(loss='mean_squared_error', optimizer=self.optimizer, metrics=[MeanSquaredError()])  # Change loss function and metrics for regression

    def fit(self, x, y):
        self.feature_names = x.columns  # Save the feature names of training data for metadata
        self.target_name = y.name  # Save the target name for metadata
        self.regressor = self._build_regressor()
        self.compile_model()
        self.regressor.fit(x, y, batch_size=self.batch_size, epochs=self.epochs)

    def predict(self, x):
        return self.regressor.predict(x)

    def _save_metadata(self, directory, regressor_filename):

        # define metadata
        self.metadata = ModelMetadata(
            addmo_class=type(self).__name__,
            addmo_commit_id=get_commit_id(),
            library=keras.__name__,
            library_model_type='Sequential',
            library_version=keras.__version__,
            target_name=self.target_name,
            features_ordered=list(self.feature_names),
            preprocessing=['We can use this to define the architecture maybe?'])

        # save metadata
        regressor_filename = os.path.splitext(regressor_filename)[0]  # Remove file extension
        metadata_path = os.path.join(directory, regressor_filename + '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata.dict(), f)

    def save_regressor(self, directory, filename=None, file_type='keras'):
        # Save model as a `.keras` file
        if filename is None:
            filename = type(self).__name__
        path = os.path.join(directory, f"{filename}.{file_type}")
        self._save_metadata(directory, filename)
        self.regressor.save(path, overwrite=True)
        print(f"Model saved to {path}")

    def load_regressor(self, regressor):
        self.regressor = regressor

    def to_scikit_learn(self, learning_rate=0.0001, epochs=10, batch_size=128, dropout=0.5):
    # Wrap the keras model to scikit in order to use Optuna Tuner

        # set default hyperparameters:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.optimizer = SGD(learning_rate=learning_rate)

        self.sklearn_regressor = KerasRegressor(
                                model=self.regressor,
                                optimizer=self.optimizer,
                                epochs=self.epochs,
                                batch_size=self.batch_size,
                                optimizer__learning_rate=self.learning_rate,
                                loss= "mean_squared_error",
                                dropout= 0.5
        )

        return self.sklearn_regressor

    def set_params(self, **params):
        # Set hyperparameters and re-compile the model with best hyperparameters returned by Optuna.

       # self.to_scikit_learn()
        self.sklearn_regressor.set_params(**params)

        # Updating keras model with new parameters.
        self.regressor = self.sklearn_regressor.model
        # Re-compile Keras model with the updated parameters.
        self.compile_model()

    def get_params(self):
        # Get hyperaparameters of the model.

        return self.regressor.get_params()


    def optuna_hyperparameter_suggest(self, trial):  # ask martin
        hyperparameters = {}

        # Suggest hyperparameters

        n_layers = trial.suggest_int("n_layers", 1, 3)
        hidden_layer_sizes = tuple(trial.suggest_int(f"n_units_l{i}", 1, 100) for i in range(n_layers))

        # Dynamic hidden layer sizes based on the number of layers
        hyperparameters["hidden_layer_sizes"] = hidden_layer_sizes

        # Other hyperparameters
        hyperparameters["activation"] = trial.suggest_categorical("activation", ["relu", "sigmoid", "tanh"])
        hyperparameters["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 1e-1)
        hyperparameters["loss"]= trial.suggest_categorical("loss", ["mse", "mae"])

        return hyperparameters

    def grid_search_hyperparameter(self):
        hyperparameter_grid = {
            "hidden_layer_sizes":  [(32,), (64,), (32, 32), (64, 64)],
            "activation": ["tanh", "relu", "softmax", "leaky_relu", "sigmoid", "exponential"],
            "optimizer": [SGD(), Adam(), RMSprop(), Adagrad()],
            "epochs": [10, 50, 100, 150],
            "learning_rate": [0.0001, 0.001, 0.01],
        }
        return hyperparameter_grid

    def default_hyperparameter(self):
        return self.sklearn_regressor.get_params()


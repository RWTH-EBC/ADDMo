import os
import json
from abc import ABC
import keras
import pandas as pd
from tensorflow.keras.models import Sequential
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

    def _build_regressor(self):
        # Add layers to model : similar to MLP
        regressor = Sequential()
        regressor.add(Dense(units=64, input_shape=(len(self.feature_names),)))
        regressor.add(Activation('relu'))
        regressor.add(Dropout(0.5))
        regressor.add(Dense(1, activation='linear'))   # Output shape = 1 for continuous variable
        return regressor

    def compile_model(self, learning_rate=0.0001, epochs=10):
        self.learning_rate = learning_rate  # Save hyperparameters for get_hyperparameters()
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

    def to_scikit_learn(self):
        pass

    def set_params(self, **params):   #ask martin if this is needed, improve code readability

        for param, value in params.items():
            if param == 'learning_rate':
                self.learning_rate = value
            elif param == 'epochs':
                self.epochs = value
            elif param == 'batch_size':
                self.batch_size = value
            elif param == 'dropout':
                # Set dropout rate for all dropout layers in the model
                for layer in self.regressor.layers:
                    if isinstance(layer, Dropout):
                        layer.rate = value
            elif param == 'optimizer':
                if value == 'SGD':
                    self.optimizer = SGD(learning_rate=self.learning_rate)
                elif value == 'Adam':
                    self.optimizer = Adam(learning_rate=self.learning_rate)
                elif value == 'RMSprop':
                    self.optimizer = RMSprop(learning_rate=self.learning_rate)
                elif value == 'Adagrad':
                    self.optimizer = Adagrad(learning_rate=self.learning_rate)
            elif param == 'activation':
                # Set activation function for all layers in the model
                for layer in self.regressor.layers:
                    if isinstance(layer, Activation):
                        layer.activation = value
            elif param == 'loss_function':
                # Example: set loss function during model compilation or define self.loss?
                self.regressor.compile(loss=value, optimizer=self.optimizer, metrics=['accuracy'])
            elif param == 'metrics':
                # Example: set evaluation metrics during model compilation
                self.regressor.compile(loss=self.regressor.loss, optimizer=self.optimizer, metrics=value)
            elif param == 'model_architecture':
                self.regressor = keras.models.model_from_json(value)


    def get_params(self):  # No built-in method

        # For Layers:
        # layer_params = []
        #     for layer in self.regressor.layers:
        #         layer_info = {
        #             'layer_type': layer.__class__.__name__,
        #             'config': layer.get_config()
        #         }
        #         layer_params.append(layer_info)

        return {
            'input_shape': self.input_shape,
            'output_shape': self.output_shape,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'dropout': 0.5,
            'optimizer': self.optimizer,    #ask: how to set parameters for different layers
            'activation': ['relu', 'softmax'],  #ask: how to set parameters for different layers
            'loss_function': 'categorical_crossentropy',
            'metrics': ['accuracy'],
            'model_architecture': self.regressor.to_json()
        }

    def optuna_hyperparameter_suggest(self, trial):  # ask martin
        hyperparameters = {}

        # Suggest hyperparameters
        n_layers = trial.suggest_int("n_layers", 1, 3)
        hidden_layer_sizes = tuple(
            trial.suggest_int(f"n_units_l{i}", 1, 100) for i in range(n_layers)
        )

        # Dynamic hidden layer sizes based on the number of layers
        hyperparameters["hidden_layer_sizes"] = hidden_layer_sizes

        # Other hyperparameters
        hyperparameters["activation"] = "relu"
        hyperparameters["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 1e-1)

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
        return self.regressor.get_params()


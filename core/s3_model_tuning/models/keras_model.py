import os
import json
from abc import ABC
import keras
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Adagrad
from core.s3_model_tuning.models.abstract_model import AbstractMLModel
from core.s3_model_tuning.models.abstract_model import ModelMetadata
from core.util.definitions import get_commit_id


class BaseKerasModel(AbstractMLModel, ABC):
    """
    Base class for Keras models.
    This class extends the AbstractMLModel, providing concrete implementations of
    common functionalities specific to keras sequential model.

    Attributes:
        model: A keras model.
    """

    def __init__(self, input_shape, output_shape):
        # Create an instance of keras model
        self.input_shape = input_shape  # Required for first layer of MLP
        self.output_shape = output_shape  # Required for final layer of MLP
        self.regressor = self.build_regressor()

    def build_regressor(self):
        # Add layers to model : similar to MLP
        regressor = Sequential()
        regressor.add(Dense(units=64, input_shape=(self.input_shape,)))
        regressor.add(Activation('relu'))
        regressor.add(Dropout(0.5))
        regressor.add(Dense(self.output_shape, activation='softmax'))
        return regressor

    def compile_model(self, learning_rate=0.0001, epochs=10):
        self.learning_rate = learning_rate  # Save hyperparameters for get_hyperparameters()
        self.epochs = epochs  # Save hyperparameters for get_hyperparameters()
        self.optimizer = SGD(learning_rate=learning_rate)  # Create an instance of SGD optimizer
        self.regressor.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=[keras.metrics.Accuracy()])

    def fit(self, x, y, epochs=10, batch_size=32, validation_data=None):
        self.x = x  # Save the training data to get feature order for metadata
        self.y = y  # Save the target column to get target name for metadata
        self.batch_size = batch_size
        self.regressor.fit(x, y, epochs=epochs, batch_size=batch_size, validation_data=validation_data)

    def predict(self, x):
        return self.regressor.predict(x)

    def _save_metadata(self, directory, regressor_filename):

        # feature names can only be extracted if x and y are pandas dataframes
        if not isinstance(self.x, pd.DataFrame):
            raise TypeError(f"x should be a pandas dataframe/series and not of type {type(self.x)}")
        if not isinstance(self.y, (pd.Series, pd.DataFrame)):
            raise TypeError(f"y should be a pandas dataframe/series and not of type {type(self.y)}")

        # define metadata
        self.metadata = ModelMetadata(
            addmo_class=type(self).__name__,
            addmo_commit_id=get_commit_id(),
            library=keras.__name__,
            library_model_type='Sequential',
            library_version=keras.__version__,
            target_name=list(self.y.columns),
            features_ordered=list(self.x.columns),
            input_shape=self.x.shape[1],  # kwargs for keras model factory
            output_shape=len(set(self.y)),  # kwargs for keras model factory
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


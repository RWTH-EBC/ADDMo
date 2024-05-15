import os
import json
from abc import ABC
import keras
import pandas as pd
import h5py
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential
from scikeras.wrappers import KerasRegressor
from tensorflow.keras.layers import Dense, Dropout, Activation
from keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Adagrad
from core.s3_model_tuning.models.abstract_model import AbstractMLModel
from core.s3_model_tuning.models.abstract_model import ModelMetadata
from core.s3_model_tuning.models.abstract_model import get_commit_id
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class BaseKerasModel(AbstractMLModel, ABC):
    """
    Base class for Keras models.
    This class extends the AbstractMLModel, providing concrete implementations of
    common functionalities specific to keras sequential model.

    Attributes:
        model: A keras model in ScikitLearn Pipeline.
    """

    # def __init__(self):
    #     # Default hyperparameters.
    #     # (ask martin: is it ok to define here, we initialise it here because it's needed for to_scikeras() ) #Todo: no, this is model specific and should be defined in the subclasses of basekeras, like in scikit learn models but its okay in the init!
    #     self.hyperparameters = {
    #         'activation': 'relu',
    #         'dropout_rate': 0.5,
    #         'optimizer': 'sgd',
    #         'learning_rate': 0.00001,
    #         'epochs': 10,
    #         'batch_size': 128,
    #     }
    #     self.epochs = self.hyperparameters['epochs']
    #     self.learning_rate = self.hyperparameters['learning_rate']
    #     self.batch_size = self.hyperparameters['batch_size']
    #     self.regressor = None
    #     # Create instance of keras model as a pipeline.
    #     self.sklearn_regressor = self._to_scikeras()

    def _save_metadata(self, directory, regressor_filename):
        # Define Metadata.
        self.metadata = ModelMetadata(
            addmo_class=type(self).__name__,
            addmo_commit_id=get_commit_id(),
            library=keras.__name__,
            library_model_type='Sequential',
            library_version=keras.__version__,
            target_name=self.target_name,
            features_ordered=list(self.feature_names),
            preprocessing=['We can use this to define the architecture maybe?'])

        # Save Metadata.
        regressor_filename = os.path.splitext(regressor_filename)[0]  # Remove file extension
        metadata_path = os.path.join(directory, regressor_filename + '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata.dict(), f)

    def save_regressor(self, directory, filename=None, file_type='h5'):
        # Save model as a `.keras` file
        if filename is None:
            filename = type(self).__name__
        path = os.path.join(directory, f"{filename}.{file_type}")
        self._save_metadata(directory, filename)
        self.regressor.save(path, overwrite=True)
        print(f"Model saved to {path}")

    def load_regressor(self, regressor):
        # Load trained model for serialisation.
        self.regressor = regressor

    # def to_scikit_learn(self, x):
    #     # Convert Keras Model to Scikit Learn Pipeline.
    #     self.input_shape = len(x.columns)
    #     self.regressor = Pipeline([
    #         ("scaler", StandardScaler()),  # Feature scaling #Todo: i think you can have the scaler directly as a layer in the model in keras (normalization layer or something like this)
    #         ("model",
    #          KerasRegressor(model=lambda: self._build_regressor(self.hyperparameters, self.input_shape),
    #                         loss='mean_squared_error',
    #                         epochs=self.epochs,
    #                         batch_size=self.batch_size, verbose=0))
    #
    #     ])
    #     return self.regressor

    # def _to_scikeras(self):
    #     # Wrap the keras model to scikit in order to use Optuna Tuner
    #     return KerasRegressor(
    #         model=lambda: self._build_regressor(self.hyperparameters),
    #         optimizer=self.hyperparameters['optimizer'],
    #         loss="mean_squared_error",
    #         epochs=self.hyperparameters['epochs'],
    #         batch_size=self.hyperparameters['batch_size'],
    #         verbose=0)





class SciKerasSequential(BaseKerasModel):
    def __init__(self):
        self.regressor = KerasRegressor() #Todo: alternatively call _build_reg
    def fit(self, x, y):
        self.feature_names = x.columns  # Save the training data to be used later for metadata #TODO This can actually go into the abstract class as function, as this is always the same, correct?
        self.target_name = y.name  # Save the target column to get target name for metadata

        self.regressor.fit(x, y, batch_size=128, epochs=self.epochs) # Todo: if possible set these params batch/epochs with the set_params function and delete here.

    def predict(self, x):
        return self.regressor.predict(x)

    def get_params(self, deep=True):
        # Get the hyperparameters of the model
        return self.regressor.get_params()

    def _build_regressor_architecture(self, hyperparameters):
        # Add layers to model.
        self.regressor = Sequential([
            Dense(64, activation=hyperparameters.get('activation', 'relu')),
            Dense(1, activation='linear'),
            ])

    def _compile_model(self, hyperparameters):
        #TODO: maybe  this works as well?
        # self.optimizer = SGD(**hyperparameters)
        optimizer = SGD(learning_rate=hyperparameters.get("learning_rate", 0.00001))  # Create an instance of SGD optimizer
        loss = hyperparameters.get("loss", MeanSquaredError())  # Create an instance of Mean Squared Error loss function
        self.regressor.compile(optimizer=optimizer, loss=loss)

    def _build_regressor(self, hyperparameters):
        self._build_regressor_architecture(hyperparameters)
        self._compile_model(hyperparameters)

    def set_params(self, hyperparameters):
        self._build_regressor(hyperparameters)
        self._compile_model(hyperparameters)
    def default_hyperparameter(self):
        return self.regressor.get_params()
    def to_scikit_learn(self):
        return self.regressor

    def optuna_hyperparameter_suggest(self, trial):
        hyperparameters = {
            "activation": trial.suggest_categorical("activation", ["relu", "sigmoid", "linear", "tanh"]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.5, step=0.1),
            "optimizer": trial.suggest_categorical("optimizer", ["sgd", "adam", "rmsprop"])
        }
        return hyperparameters

    def grid_search_hyperparameter(self):
        hyperparameter_grid = {
            "hidden_layer_sizes": [(32,), (64,), (32, 32), (64, 64)],
            "activation": ["tanh", "relu", "softmax", "leaky_relu", "sigmoid", "exponential"],
            "optimizer": [SGD(), Adam(), RMSprop(), Adagrad()],
            "epochs": [10, 50, 100, 150],
            "learning_rate": [0.0001, 0.001, 0.01],
        }
        return hyperparameter_grid



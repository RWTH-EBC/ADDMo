import keras
import tensorflow as tf
import onnx
import numpy as np
from abc import ABC
from packaging import version

from keras.src.models.sequential import Sequential
from keras.src.layers import Input, Dense, Normalization, Activation
from keras.src.losses import MeanSquaredError
from keras.src.callbacks import EarlyStopping
from scikeras.wrappers import KerasRegressor
from addmo.s3_model_tuning.models.abstract_model import AbstractMLModel
from addmo.s3_model_tuning.models.abstract_model import ModelMetadata


class BaseKerasModel(AbstractMLModel, ABC):
    """
    Base class for Keras models.
    """

    def _define_metadata(self):
        """
        Define metadata.
        """
        self.metadata = ModelMetadata(
            addmo_class=type(self).__name__,
            addmo_commit_id=ModelMetadata.get_commit_id(),
            library=keras.__name__,
            library_model_type=type(self.regressor.model).__name__,
            library_version=keras.__version__,
            target_name=self.y_fit.name,
            features_ordered=list(self.x_fit.columns),
            preprocessing=['Scaling as layer of the ANN.'])


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
        self.regressor = self.to_scikit_learn(x)
        self.regressor.fit(x.values.astype(np.float32), y.values.astype(np.float32))

    def predict(self, x):
        return self.regressor.predict(x.values.astype(np.float32))

    def get_params(self, deep=True):
        """
        Get the hyperparameters of the model.
        """
        # get scikeras params
        params = self.regressor.get_params(deep=deep)
        # additional params not covered by scikeras (update only if not present)
        for key, value in self.hyperparameters.items():
            if key not in params:
                params[key] = value

        return params

    def set_params(self, hyperparameters):
        """""
        Update the hyperparameters in internal storage, which is accessed while building the
        regressor. Not done here, because compilation requires the input_shape to be available.
        """
        for key, value in hyperparameters.items():
            self.hyperparameters[key] = value

    def _save_regressor(self, path, file_type):
        """
        Save regressor as a .keras or .onnx file.
        """

        if file_type == 'keras':
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

    def load_regressor(self, regressor, input_shape):
        """""
        Load trained model for serialisation.
        """
        # Create dummy system_data for initialization of loaded model
        x = np.zeros((1, input_shape))
        y = np.zeros((1,))
        self.regressor = KerasRegressor(regressor)
        # Initialize model to avoid re-fitting
        self.regressor.initialize(x, y)

    def _build_regressor_architecture(self, input_shape):
        """
        Builds a sequential model.
        """
        sequential_regressor = Sequential()
        # normalizer = Normalization(axis=-1)  # Preprocessing layer
        sequential_regressor.add(Input(shape=input_shape))
        # sequential_regressor.add(normalizer)
        # Adding hidden layers based on hyperparameters
        for units in self.hyperparameters['hidden_layer_sizes']:
            sequential_regressor.add(Dense(units=units, activation='relu'))

        sequential_regressor.add(Dense(1, activation='linear'))
        return sequential_regressor

    def _build_regressor(self, x):
        """""
        Returns a compiled sequential model.
        """
        input_shape = (len(x.columns),)
        sequential_regressor = self._build_regressor_architecture(input_shape)

        # Normalisation of first layer (input system_data).
        # sequential_regressor.layers[0].adapt(x.to_numpy())  # Normalisation initialisation works only on np arrays

        # define optimizer explicitly to avoid user warnings that optimizer could not be loaded
        optimizer = tf.keras.optimizers.RMSprop()
        optimizer.build(sequential_regressor.trainable_variables)

        sequential_regressor.compile(loss=self.hyperparameters['loss'], optimizer=optimizer)
        return sequential_regressor

    def to_scikit_learn(self, x):
        """""
        Convert Keras Model to Scikeras Regressor for tuning.
        """

        regressor_scikit = KerasRegressor(model=self._build_regressor(x),
                                          batch_size=self.hyperparameters['batch_size'],
                                          loss=self.hyperparameters['loss'],
                                          epochs=self.hyperparameters['epochs'],
                                          verbose=0,
                                          callbacks=self.hyperparameters['callbacks'])
        return regressor_scikit

    def default_hyperparameter(self):
        """"
        Return default hyperparameters.
        """
        regressor = KerasRegressor()
        hyperparameters = regressor.get_params()
        # Define default loss if not present
        if hyperparameters['loss'] is None:
            hyperparameters['loss'] = MeanSquaredError()
        hyperparameters['hidden_layer_sizes'] = [16]
        hyperparameters['batch_size'] = 200
        hyperparameters['epochs'] = 1000
        hyperparameters['callbacks'] = [EarlyStopping(monitor='loss',
                                                      min_delta=0.0001,
                                                      verbose=1,
                                                      patience=10)]

        return hyperparameters

    def optuna_hyperparameter_suggest(self, trial):

        n_layers = trial.suggest_int("n_layers", 1, 2)
        hidden_layer_sizes = tuple(trial.suggest_int(f"n_units_l{i}", 1, 1000) for i in range(1, n_layers + 1, 1))
        hyperparameters = {
            "hidden_layer_sizes": hidden_layer_sizes,
        }
        return hyperparameters

    def grid_search_hyperparameter(self):

        hyperparameter_grid = {
            "hidden_layer_sizes": [(64,), (128, 64), (256, 128, 64)],
            "loss": [MeanSquaredError()],
            "epochs": [50]
        }
        return hyperparameter_grid

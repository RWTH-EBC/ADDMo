import math
import os
import random
import re
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from keras import Sequential, models
from keras.engine.base_layer import Layer
from keras.layers import Dense, Rescaling, BatchNormalization

from machine_learning_util.data_handling import TrainingData, write_pkl, read_pkl


class TunerLayer(ABC):
    """Abstract ANN layer for tuning"""

    def __init__(self, optional: bool = True):

        self.optional = optional

    @abstractmethod
    def build_keras_layer(self) -> Layer:
        """Returns parameterized keras layer"""
        pass


class TunerRescaling(TunerLayer):
    """Rescaling layer for tuning"""

    def __init__(self, scale: float, offset: float, optional: bool = False):

        super().__init__(optional=optional)

        self.scale = scale
        self.offset = offset

    def build_keras_layer(self) -> Layer:
        """Returns paramterized keras rescaling layer

        Returns
        -------
        Rescaling
            paramterized keras rescaling layer
        """
        return Rescaling(scale=self.scale, offset=self.offset)


class TunerBatchNormalizing(TunerLayer):
    """Batch normalizing layer for tuning"""

    def __init__(self, axis: int = 1, optional: bool = False):

        super().__init__(optional=optional)

        self.axis = axis

    def build_keras_layer(self) -> Layer:
        """Returns paramterized keras batch normalizing layer

        Returns
        -------
        BatchNormalizing
            paramterized keras batch normalizing layer
        """
        return BatchNormalization(axis=self.axis)


class TunerDense(TunerLayer):
    """Dense layer for tuning"""
    def __init__(self, units: tuple = None, activations: tuple = None, optional: bool = False):
        super().__init__(optional=optional)

        # default of 8 or 16 neurons
        if units is None:
            units = (8, 16,)

        # default activation is sigmoid
        if activations is None:
            activations = ('sigmoid',)

        self.units = units
        self.activations = activations

    def build_keras_layer(self) -> Dense:
        """Returns paramterized dense layer

        Returns
        -------
        Dense
            paramterized keras dense layer
        """
        units = random.choice(self.units)
        activation = random.choice(self.activations)

        return Dense(units=units, activation=activation)


class TunerModel:
    """
    Blueprint for a Sequential Keras model.
    Layers can be made optional by setting optional to True.
    """

    def __init__(
            self,
            *layers: TunerLayer,
            name: str,
            optimizer: str = 'adam',
            loss: str = 'mse',
    ):
        """

        Parameters
        ----------
        layers: *TunerLayer
            list of tuner layers
        name: str
            name of the model
        optimizer: str
            name of the optimizer to be used
        loss: str
            name of the loss function to be used
        """
        self.layers = layers
        self.optimizer = optimizer
        self.loss = loss

        # check if the name is valid
        assert re.match("^[A-Za-z0-9_-]*$", name), 'Please make sure you set a valid TunerModel name.' \
                                                   'Allowed characters: A-Z, a-z, 0-9, _, -'
        self.name = name

    def build_sequential(self, name: str = None) -> Sequential:
        """Build (random) keras model from specification

        Parameters
        ----------
        name: str
            name of the model

        Returns
        -------
        Sequential
            (random) keras sequential model from specification
        """

        # create random ann id
        ann_id = random.randint(1000, 9999)

        # ann name
        if name is None:
            name = f'{self.name}_{ann_id}'

        # create sequential keras model
        keras_model = Sequential(name=name)

        # iterate over every layer
        for layer in self.layers:

            # check if the layer is optional and maybe skip it
            if layer.optional and random.getrandbits(1):
                continue

            # add layer
            keras_model.add(layer.build_keras_layer())

        # output layer
        keras_model.add(Dense(units=1, activation='linear'))

        # compile the model
        keras_model.compile(optimizer=self.optimizer, loss=self.loss)

        return keras_model


class NeuralNetwork:
    """Representation of a neural network"""

    def __init__(
            self,
            directory:  str = 'Models',
    ):
        self.sequential = None
        self.name = None
        self.directory: str = directory

    def build_sequential(self, tuner_model: TunerModel):
        """ Builds a random sequential keras model by using the tuner model. """

        self.delete_sequential()

        # builds a new random sequential ann
        self.sequential = tuner_model.build_sequential()

        # update the name
        self.name = self.sequential.name

    def fit(self, training_data: TrainingData, **kwargs):
        """ trains the ann on data """

        assert self.sequential is not None, 'Please call build_sequential() before fitting.'

        self.sequential.fit(
            x=training_data.xTrain.astype(np.float32),
            y=training_data.yTrain.astype(np.float32),
            validation_data=(training_data.xValid.astype(np.float32), training_data.yValid.astype(np.float32)),
            **kwargs,
        )

    def save_sequential(self, override: bool = True):
        """ saves the sequential model and deletes it to later safely pickle this instance. """

        assert self.sequential is not None, 'Please make sure to build a sequential before saving.'

        # save the sequential to the disc
        self.sequential.save(self._sequential_filepath)

        # before pickeling the sequential ann and casadi ann must be deleted
        del self.sequential

        # now pickle the NeuralNetwork object to the disc
        write_pkl(self, self.name, self._sequential_filepath, override)

    def delete_sequential(self):
        """ deletes the sequential keras model completely """

        del self.sequential

        # remove sequential model from disc
        if self.name is not None:
            os.remove(self._sequential_filepath)

    @property
    def _sequential_filepath(self) -> str:
        """ location at which the sequential keras model is saved. """

        assert self.name is not None, 'Sequential model can not be deleted. Please first build one.'

        return f'{self.directory}//{self.name}'

    def test(self, training_data: TrainingData, metric: str = 'rmse') -> float:
        """ Tests the ann on a given test dataset and returns the score """

        x_test = training_data.xTest
        y_test = training_data.yTest

        errors = self.prediction_error(x_test, y_test, metric=metric)

        score = errors.mean()

        if metric == 'rmse':
            score = math.sqrt(score)

        return score

    def prediction_error(self, xTest: np.ndarray, yTest: np.ndarray, metric: str = 'mae') -> np.ndarray:
        """ Tests the ann on given dataset and returns the score per datapoint """
        assert self.sequential is not None, 'Please call build_sequential() before fitting.'
        assert self.sequential.built, 'Please call fit() before testing.'
        assert metric in ['mse', 'mae', 'me', 'rmse']

        df = pd.DataFrame()

        df['y_real'] = yTest.squeeze()

        df['y_pred'] = self.sequential.predict(xTest.astype(np.float32)).squeeze()

        if metric == 'mae':
            df['error'] = abs(df['y_pred'] - df['y_real'])
        elif metric == 'me':
            df['error'] = df['y_pred'] - df['y_real']
        elif metric == 'mse' or metric == 'rmse':
            df['error'] = (df['y_pred'] - df['y_real']) ** 2
        else:
            raise ValueError('Please select a proper metric.')

        return df['error']

    def predict(self, xTest: np.ndarray) -> np.ndarray:
        """ Predicts a given test datapoint """
        assert self.sequential is not None, 'Please call build_sequential() before fitting.'
        assert self.sequential.built, 'Please call fit() before testing.'

        return self.sequential.predict(xTest.astype(np.float32), verbose=0).squeeze()

    def load_sequential(self, filepath: str = None):
        """Loads a sequential keras model from the given filepath."""

        if filepath is None:
            filepath = self._sequential_filepath

        # load Sequential from the disc
        self.sequential = models.load_model(filepath=filepath)

        # make sure the name is correctly loaded
        self.name = self.sequential.name


class NetworkTrainer:
    """Hyperparamter tuning for neural networks"""

    def __init__(
            self,
            directory:  str = 'stored_data\\Predictors',
            metric: str = 'rmse'
    ):

        self.directory:         str = directory
        self.neural_networks:   list[NeuralNetwork] = list()
        self.metric: str = metric

    def build(self, n: int, keras_tuner: TunerModel):
        """ Builds n random Sequential Keras models """

        for i in range(n):
            neural_network = NeuralNetwork(directory=self.directory + '\\Models')
            neural_network.build_sequential(keras_tuner)

            self.neural_networks.append(neural_network)

    def fit(self, training_data: TrainingData, **kwargs):
        """ trains all sequential Keras models that are stored in neural_networks """

        assert len(self.neural_networks) != 0, 'Make sure to call build() first.'

        for neural_network in self.neural_networks:
            neural_network.fit(
                training_data=training_data,
                **kwargs
            )

        self.sort(training_data)

    def sort(self, training_data: TrainingData):
        """ sorts all sequential Keras models that are stored in neural_networks based on their score """

        scores = dict()

        # calculate the score fore every neural network
        for neural_network in self.neural_networks:
            scores[neural_network] = neural_network.test(metric=self.metric, training_data=training_data)

        # sort neural networks
        keys = list(scores.keys())
        values = list(scores.values())
        sorted_value_index = np.argsort(values)
        self.neural_networks = [keys[i] for i in sorted_value_index]

    def save(self, filename: str, override: bool = False):
        """ saves all neural networks to the disc """

        self.neural_networks = [self.best]

        for neural_network in self.neural_networks:

            neural_network.save_sequential()

        write_pkl(self, filename, self.directory, override)

    @property
    def best(self) -> NeuralNetwork:
        """ Returns the NeuralNetwork with the best score """
        return self.neural_networks[0]


def load_network_trainer(filename: str, directory: str = 'stored_data\\Predictors') -> NetworkTrainer:

    # read from disc
    network_trainer = read_pkl(filename, directory)

    # make sure the loaded data is a NetworkTrainer
    assert isinstance(network_trainer, NetworkTrainer), \
        f'Wrong type loaded. File at {directory}//{filename} is not from type NeuralNetwork'

    for neural_network in network_trainer.neural_networks:
        neural_network: NeuralNetwork
        neural_network.load_sequential()

    return network_trainer

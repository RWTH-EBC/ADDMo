import numpy as np
from keras.callbacks import EarlyStopping

import extrapolation_detection.machine_learning_util.data_handling as dh
from extrapolation_detection.machine_learning_util.ann import NetworkTrainer, load_network_trainer, TunerModel


def train_ann(name: str, tuner: TunerModel, training_interval: list, val_fraction: float = 0.1,
              test_fraction: float = 0.1, shuffle: bool = True, n: int = 20, epochs: int = 10000,
              batch_size: int = 100, callbacks=[EarlyStopping(patience=1000, verbose=1, restore_best_weights=True)],
              metric: str = 'rmse'):
    """ Trains ANN

    Parameters
    ----------
    name: str
        name of the ANN
    tuner: TunerModel
        tuner model for ANN creation
    training_interval: list
        list of indices to be used as training, validation and test split
    val_fraction: float
        relative fraction of data points from dataIndices to be used as validation split
    test_fraction: float
            relative fraction of data points from dataIndices to be used as test split
    shuffle: bool
        If set to True, training, validation and test datapoints will be shuffled
    n: int
        number of ANNs to be trained
    epochs: int
        number of training epochs
    batch_size: int
        batch size for ANN training
    callbacks: list
        list of callbacks for ANN training
    metric: str
        evaluation metric
    """

    # Load data
    data = dh.load_csv(name, path='data')

    # Preprocess data
    data = dh.split_simulation_data(data, training_interval, val_fraction, test_fraction, random_state=1, shuffle=shuffle)
    dh.write_pkl(data, 'data', directory=name, override=False)

    # Specify ANN Training
    trainer = NetworkTrainer(directory=name, metric=metric)
    trainer.build(n=n, keras_tuner=tuner)

    # Fit ANN to training data
    trainer.fit(training_data=data['available_data'], epochs=epochs, batch_size=batch_size, verbose=1,
                callbacks=callbacks)
    trainer.save(name, override=False)
    trainer = load_network_trainer(name, name)

    # Score ANN on data
    errors = dict()
    errors['train_error'] = trainer.best.prediction_error(data['available_data'].xTrain, data['available_data'].yTrain,
                                                          metric='mae')
    errors['val_error'] = trainer.best.prediction_error(data['available_data'].xValid, data['available_data'].yValid,
                                                        metric='mae')
    errors['test_error'] = trainer.best.prediction_error(data['available_data'].xTest, data['available_data'].yTest,
                                                         metric='mae')
    dh.write_pkl(errors, 'errors', directory=name, override=False)
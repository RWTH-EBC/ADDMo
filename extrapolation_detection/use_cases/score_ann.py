from typing import Callable

import numpy as np

import machine_learning_util.ann
import machine_learning_util.data_handling as dh


def score(name: str):
    """ Scores ANN with remaining data

    Parameters
    ----------
    name: str
        name of the ANN
    """

    # Load data
    ann = machine_learning_util.ann.load_network_trainer(name, name).best
    data: dict = dh.read_pkl('data', name)

    # Score
    score_dct = dict()
    score_dct['errors'] = ann.prediction_error(data['x_data'], data['y_data'], metric='mae')
    dh.write_pkl(score_dct, 'data_error', name, override=False)


def score_2D(name: str, model: Callable, contour_detail_error: int = 100):
    """ Scores ANN for 2D plot

    Parameters
    ----------
    name: str
        name of the ANN
    model: Callable
        Callback of the underlying model to be evaluated
    contour_detail_error: int
        details of the validity domain curve
    """

    # Load data
    ann = machine_learning_util.ann.load_network_trainer(name, name).best
    data: dict = dh.read_pkl('data', name)

    # Get bounds of 2D plot
    left = np.amin(np.concatenate((data['TrainingData'].xTrain, data['TrainingData'].xValid,
                                   data['TrainingData'].xTest, data['x_data']))[:, 0])
    right = np.amax(np.concatenate((data['TrainingData'].xTrain, data['TrainingData'].xValid,
                                    data['TrainingData'].xTest, data['x_data']))[:, 0])
    bottom = np.amin(np.concatenate((data['TrainingData'].xTrain, data['TrainingData'].xValid,
                                     data['TrainingData'].xTest, data['x_data']))[:, 1])
    top = np.amax(np.concatenate((data['TrainingData'].xTrain, data['TrainingData'].xValid,
                                  data['TrainingData'].xTest, data['x_data']))[:, 1])

    # Generate Meshgrid
    xspace = np.linspace(left, right, contour_detail_error)
    yspace = np.linspace(bottom, top, contour_detail_error)
    xx, yy = np.meshgrid(xspace, yspace)

    score_2D_dct = dict()
    score_2D_dct['xx_error'] = xx
    score_2D_dct['yy_error'] = yy
    contour_error = np.zeros((contour_detail_error, contour_detail_error))
    # Evaluate meshgrid with ANN and underlying model
    for i in range(0, contour_detail_error):
        for j in range(0, contour_detail_error):
            contour_error[i, j] = np.sqrt(
                np.array(ann.predict(np.array([xx[i, j], yy[i, j]]).reshape(1, -1)) - model(xx[i, j], yy[i, j])) ** 2)
    score_2D_dct['contour_error'] = contour_error

    dh.write_pkl(score_2D_dct, 'errors_2D', name, override=False)


def carnot_model(t_amb: float, p_el: float, supply_temp: float = 40) -> float:
    """ Model of a simplified heatpump with carnot efficiency

    Parameters
    ----------
    t_amb: float
        ambient temperature
    p_el: float
        electric power
    supply_temp: float
        supply temperature

    Returns
    -------
    float
        heat flow
    """
    return p_el * (273.15 + supply_temp) / (supply_temp - t_amb)

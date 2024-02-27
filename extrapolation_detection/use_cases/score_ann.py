from typing import Callable

import numpy as np

from extrapolation_detection import machine_learning_util
import extrapolation_detection.machine_learning_util.data_handling as dh


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
    score_dct['errors'] = ann.prediction_error(data['non_available_data'].x_remaining, data['non_available_data'].y_remaining, metric='mae')
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
    left = np.amin(np.concatenate((data['available_data'].xTrain, data['available_data'].xValid,
                                   data['available_data'].xTest, data['non_available_data'].x_remaining))[:, 0])
    right = np.amax(np.concatenate((data['available_data'].xTrain, data['available_data'].xValid,
                                    data['available_data'].xTest, data['non_available_data'].x_remaining))[:, 0])
    bottom = np.amin(np.concatenate((data['available_data'].xTrain, data['available_data'].xValid,
                                     data['available_data'].xTest, data['non_available_data'].x_remaining))[:, 1])
    top = np.amax(np.concatenate((data['available_data'].xTrain, data['available_data'].xValid,
                                  data['available_data'].xTest, data['non_available_data'].x_remaining))[:, 1])

    # Generate Meshgrid
    xspace = np.linspace(left, right, contour_detail_error)
    yspace = np.linspace(bottom, top, contour_detail_error)
    xx, yy = np.meshgrid(xspace, yspace)

    score_2D_dct = dict()
    score_2D_dct['var1_meshgrid'] = xx
    score_2D_dct['var2_meshgrid'] = yy
    error_on_mesh = np.zeros((contour_detail_error, contour_detail_error))
    # Evaluate meshgrid with ANN and underlying model
    for i in range(0, contour_detail_error):
        for j in range(0, contour_detail_error):
            error_on_mesh[i, j] = np.sqrt(
                np.array(ann.predict(np.array([xx[i, j], yy[i, j]]).reshape(1, -1)) - model(xx[i, j], yy[i, j])) ** 2)
    score_2D_dct['error_on_mesh'] = error_on_mesh

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

from typing import Callable

import numpy as np

import extrapolation_detection.machine_learning_util.data_handling as dh
from extrapolation_detection.machine_learning_util.util import rearrange_training_data


def train_clf(name: str, clf_name: str, clf_callback: Callable, outlier_fraction: float, score: str = 'fbeta',
              beta: float = 1, use_train_for_validation: bool = False):
    """ Trains classifier

    Parameters
    ----------
    name: str
        name of the ANN
    clf_name: str
        name of the classifier
    clf_callback: Callable
        hyperparameter tuning callback for classifier creation
    outlier_fraction: float
        fraction of (expected) outliers
    score: str
        score to be evaluated (from detector/scoring.py)
    beta: float
        beta value for F-score
    use_train_for_validation:
        If True, training data will also be used for validation (in addition to validation data)
    """

    # Load data
    data: dict = dh.read_pkl('data', name)
    validity_domain: dict = dh.read_pkl('true_validity_classified_train_test_val', name)

    # Prepare data
    x_train = data['available_data'].xTrain
    x_val = data['available_data'].xValid
    x_test = data['available_data'].xTest
    x_val = np.concatenate((x_val, x_test))
    y_train = validity_domain['ground_truth_train']
    y_val = validity_domain['ground_truth_val']
    y_test = validity_domain['ground_truth_test']
    y_val = np.concatenate((y_val, y_test))

    # Preprocess data
    x_train, x_val, y_train, y_val = rearrange_training_data(x_train, x_val, y_train, y_val)

    # Tune classifier
    hyper = clf_callback(outlier_fraction, score_name=score, beta=beta)
    if use_train_for_validation:
        y_train = y_train.reshape((-1, 1))
    else:
        y_train = None
    clf, threshold = hyper.get_clf(x_train, x_val, y_val.reshape((-1, 1)), groundtruth_train=y_train)

    # Train classifier
    clf.train(x_train)

    # Save
    dh.write_pkl(clf, clf_name, name, override=False)
    dh.write_pkl(threshold, clf_name + '_threshold', name, override=False)


def train_clf_ideal(name, clf_name, clf_callback, outlier_fraction, score='fbeta', beta=1):
    """ Trains (ideal) classifier

    Parameters
    ----------
    name: str
        name of the ANN
    clf_name: str
        name of the classifier
    clf_callback: Callable
        hyperparameter tuning callback for classifier creation
    outlier_fraction: float
        fraction of (expected) outliers
    score: str
        score to be evaluated (from detector/scoring.py)
    beta: float
        beta value for F-score
    """

    # Load data
    data: dict = dh.read_pkl('data', name)
    validity_domain: dict = dh.read_pkl('true_validity_classified_train_test_val', name)

    # Prepare data
    x_train = data['available_data'].xTrain
    x_val = data['available_data'].xValid
    x_test = data['available_data'].xTest
    x_remaining = data['non_available_data'].x_remaining
    x_val = np.concatenate((x_val, x_test, x_remaining))
    y_train = validity_domain['ground_truth_train']
    y_val = validity_domain['ground_truth_val']
    y_test = validity_domain['ground_truth_test']
    y_data = dh.read_pkl('true_validity_classified_remaining', name)
    y_val = np.concatenate((y_val, y_test, y_data))

    # Preprocess data
    x_train, x_val, y_train, y_val = rearrange_training_data(x_train, x_val, y_train, y_val)

    # Tune classifier
    hyper = clf_callback(outlier_fraction, score_name=score, beta=beta)
    clf, threshold = hyper.get_clf(x_train, x_val, y_val.reshape((-1, 1)))

    # Train classifier
    clf.train(x_train)

    # Save
    dh.write_pkl(clf, clf_name + '_ideal', name, override=False)
    dh.write_pkl(threshold, clf_name + '_ideal_threshold', name, override=False)

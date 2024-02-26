import math

import numpy as np

import extrapolation_detection.machine_learning_util.data_handling as dh
from extrapolation_detection.detector.detector import D_KNN, D_ParzenWindow, D_OCSVM, D_GP, D_IF
from extrapolation_detection.machine_learning_util.util import rearrange_training_data


def train_clf_untuned(name: str, clf_name: str, clf: D_KNN or D_ParzenWindow or D_OCSVM or D_GP or D_IF,
                      outlier_fraction: float):
    """ Trains classifier without hyperparameter tuning

    Parameters
    ----------
    name: str
        name of the ANN
    clf_name: str
        name of the classifier
    clf: D_KNN or D_ParzenWindow or D_OCSVM or D_GP or D_IF
        instance of classifier to be used
    outlier_fraction: float
        fraction of (expected) outliers
    """

    # Load data
    data: dict = dh.read_pkl('data', name)
    validity_domain: dict = dh.read_pkl('validity_domain', name)

    # Prepare data
    x_train = data['TrainingData'].xTrain
    x_val = data['TrainingData'].xValid
    x_test = data['TrainingData'].xTest
    x_val = np.concatenate((x_val, x_test))
    y_train = validity_domain['ground_truth_train']
    y_val = validity_domain['ground_truth_val']
    y_test = validity_domain['ground_truth_test']
    y_val = np.concatenate((y_val, y_test))

    # Preprocess data
    x_train, x_val, y_train, y_val = rearrange_training_data(x_train, x_val, y_train, y_val)

    # Tune classifier
    clf.train(x_train)

    # Calculate decision threshold
    scores_train = clf.get_decision_scores()
    scores_valid = clf.score(x_val)
    scores = np.concatenate((scores_train, scores_valid))
    scores_sorted = np.sort(scores, axis=0)
    threshold = scores_sorted[math.floor((1 - outlier_fraction) * len(scores_sorted))]

    # Save
    dh.write_pkl(clf, clf_name, name, override=False)
    dh.write_pkl(threshold, clf_name + '_threshold', name, override=False)

import numpy as np
import pandas as pd

import extrapolation_detection.machine_learning_util.data_handling as dh
from extrapolation_detection.machine_learning_util.util import rearrange_training_data
from extrapolation_detection.detector.detector_factory import DetectorFactory
from extrapolation_detection.detector.config.detector_config import DetectorConfig

def tune_detector(detector_name: str, x_train, x_val, y_val, config_detector: \
    DetectorConfig):
    """ Trains classifier"""

    # get detector tuner
    detector_factory = DetectorFactory()
    detector_tuner = detector_factory.detector_tuner_factory(detector_name)


    # set data of the detector
    detector_tuner.outlier_fraction=config_detector.outlier_fraction
    detector_tuner.beta=config_detector.beta_f_score

    # since I am still using the detector optimization code from Patricks MA we need to switch to
    # ndarrays
    x_train = x_train.values
    x_val = x_val.values
    y_val = y_val.values.reshape(-1, 1)

    # tune detector and outlier threshold
    clf, threshold = detector_tuner.get_clf(x_train, x_val, y_val)
    clf.train(x_train)

    # to dataframe for saving in human readable csv format
    threshold = pd.DataFrame([threshold])

    return clf, threshold


def train_clf_untuned(x_train, outlier_fraction: float):
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

    # Tune classifier
    clf.train(x_train)

    # Calculate decision threshold which is normally tuned by the hyperparameter tuning
    scores_train = clf.get_decision_scores()
    scores_valid = clf.score(x_val)
    scores = np.concatenate((scores_train, scores_valid))
    scores_sorted = np.sort(scores, axis=0)
    threshold = scores_sorted[math.floor((1 - outlier_fraction) * len(scores_sorted))]

    # Save
    dh.write_pkl(clf, clf_name, name, override=False)
    dh.write_pkl(threshold, clf_name + '_threshold', name, override=False)
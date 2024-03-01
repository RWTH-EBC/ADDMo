import numpy as np
import pandas as pd

from extrapolation_detection.detector.detector_factory import DetectorFactory
from extrapolation_detection.detector.config.detector_config import DetectorConfig
from extrapolation_detection.n_D_extrapolation.true_validity_domain import infer_threshold

def tune_detector(detector_name: str, x_train, x_val, y_val, config_detector: \
    DetectorConfig):
    """ tunes detector and its novelty score threshold"""

    # get detector tuner
    detector_factory = DetectorFactory()
    detector_tuner = detector_factory.detector_tuner_factory(detector_name)


    # set data of the detector
    detector_tuner.beta=config_detector.beta_f_score

    # since I am still using the detector optimization code from Patricks MA we need to switch to
    # ndarrays
    x_train = x_train.values
    x_val = x_val.values
    y_val = y_val.values.reshape(-1, 1)

    # tune detector and novelty score threshold
    detector, nd_threshold = detector_tuner.get_clf(x_train, x_val, y_val)
    detector.train(x_train)

    # to dataframe for saving in human readable csv format
    nd_threshold = pd.DataFrame([nd_threshold])

    return detector, nd_threshold


def untuned_detector(detector_name: str, x_train, x_val, outlier_fraction):
    """ Trains classifier without hyperparameter tuning but with a fixed ideal threshold"""

    # get detector tuner
    detector_factory = DetectorFactory()
    detector = detector_factory.detector_factory(detector_name)

    # since I am still using the detector optimization code from Patricks MA we need to switch to
    # ndarrays
    x_train = x_train.values
    x_val = x_val.values

    # Tune classifier
    detector.train(x_train)

    # Calculate decision threshold which is normally tuned by the hyperparameter tuning
    scores_train = detector.get_decision_scores()
    scores_valid = detector.score(x_val)
    scores = np.concatenate((scores_train, scores_valid))

    # get novelty detection score threshold
    scores = pd.Series(scores)
    nd_threshold = infer_threshold(outlier_fraction, scores)

    # to dataframe for saving in human readable csv format
    nd_threshold = pd.DataFrame([nd_threshold])

    return detector, nd_threshold


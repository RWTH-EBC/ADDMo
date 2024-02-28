import numpy as np

import extrapolation_detection.machine_learning_util.data_handling as dh
from extrapolation_detection.machine_learning_util.util import rearrange_training_data
from extrapolation_detection.detector.detector_factory import DetectorFactory
from extrapolation_detection.detector.config.detector_config import DetectorConfig

def tune_detector(detector_name: str, x_train, x_val, y_train, y_val, config_detector: \
    DetectorConfig):
    """ Trains classifier"""

    # get detector tuner
    detector_factory = DetectorFactory()
    detector_tuner = detector_factory.detector_tuner_factory(detector_name)


    # set data of the detector
    detector_tuner(outlier_fraction=config_detector.outlier_fraction, beta=config_detector.beta)


    # tune detector and outlier threshold
    clf, threshold = detector_tuner.get_clf(x_train, x_val, y_val.reshape((-1, 1)), groundtruth_train=y_train)
    clf.train(x_train)

    return clf, threshold



def train_clf_ideal(name, clf_name, clf_callback, outlier_fraction, score='fbeta', beta=1):


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
import math

import numpy as np

import extrapolation_detection.machine_learning_util.data_handling as dh
from extrapolation_detection.detector import scoring


def true_validity_classified_train_val_test(name: str, outlier_threshold: float, threshold_is_fraction: bool=True):
    """ Specifies validity domain of regressor for training, validation and test data

    outlier_threshold: float
        fraction of outliers in percent or absolute error in respective error metric
    threshold_absolute: bool
        whether threshold is given as absolute error or a fraction of outliers to calculate the absolute error is given
    """

    # Read data
    errors: dict = dh.read_pkl('errors', name)

    validity_domain_dct = dict()

    # Calculate error threshold
    if threshold_is_fraction:
        tvt_error = np.concatenate((errors['val_error'], errors['test_error'], errors['train_error']))
        tvt_error_sorted = np.sort(tvt_error, axis=0)
        validity_domain_dct['error_threshold'] = \
            tvt_error_sorted[math.floor((1 - outlier_threshold) * len(tvt_error_sorted))]
    else:
        validity_domain_dct['error_threshold'] = outlier_threshold

    # Calculate groundtruth training
    ground_truth_train = np.zeros(len(errors['train_error']))
    ground_truth_train[errors['train_error'] > validity_domain_dct['error_threshold']] = 1
    validity_domain_dct['ground_truth_train'] = ground_truth_train

    # Calculate groundtruth validation
    ground_truth_val = np.zeros(len(errors['val_error']))
    ground_truth_val[errors['val_error'] > validity_domain_dct['error_threshold']] = 1
    validity_domain_dct['ground_truth_val'] = ground_truth_val

    # Calculate groundtruth test
    ground_truth_test = np.zeros(len(errors['test_error']))
    ground_truth_test[errors['test_error'] > validity_domain_dct['error_threshold']] = 1
    validity_domain_dct['ground_truth_test'] = ground_truth_test

    # Save
    dh.write_pkl(validity_domain_dct, 'true_validity_classified_train_test_val', name, override=False)


def true_validity_classified_remaining(name: str):
    """ Evaluates validity domain groundtruth for remaining data
    """

    # Load data
    score_dct: dict = dh.read_pkl('data_error', name)
    validity_domain_dct: dict = dh.read_pkl('true_validity_classified_train_test_val', name)

    # Calculate grountruth
    true_validity_classified_remaining = np.zeros(len(score_dct['errors']))
    true_validity_classified_remaining[score_dct['errors'] > validity_domain_dct['error_threshold']] = 1

    # Calculate outlier share
    outlier_share = scoring.get_outlier_share(score_dct['errors'], validity_domain_dct['error_threshold'])

    # Save
    dh.write_pkl(true_validity_classified_remaining, 'true_validity_classified_remaining', name, override=False)
    dh.write_pkl(outlier_share, 'outlier_share', name, override=False)

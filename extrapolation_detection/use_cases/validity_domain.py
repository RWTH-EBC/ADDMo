import math

import numpy as np

import machine_learning_util.data_handling as dh
from detector import scoring


def validity_domain(name: str, outlier_threshold: float, threshold_is_fraction: bool=True):
    """ Specifies validity domain of ANN

    Parameters
    ----------
    name: ANN
        name of the ANN
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

    # Calculate grountruth training
    ground_truth_train = np.zeros(len(errors['train_error']))
    ground_truth_train[errors['train_error'] > validity_domain_dct['error_threshold']] = 1
    validity_domain_dct['ground_truth_train'] = ground_truth_train

    # Calculate grountruth validation
    ground_truth_val = np.zeros(len(errors['val_error']))
    ground_truth_val[errors['val_error'] > validity_domain_dct['error_threshold']] = 1
    validity_domain_dct['ground_truth_val'] = ground_truth_val

    # Calculate grountruth test
    ground_truth_test = np.zeros(len(errors['test_error']))
    ground_truth_test[errors['test_error'] > validity_domain_dct['error_threshold']] = 1
    validity_domain_dct['ground_truth_test'] = ground_truth_test

    # Save
    dh.write_pkl(validity_domain_dct, 'validity_domain', name, override=False)


def validity_domain_data(name: str):
    """ Evaluates validity domain groundtruth for remaining data

    Parameters
    ----------
    name: str
        name of the ANN
    """

    # Load data
    score_dct: dict = dh.read_pkl('data_error', name)
    validity_domain_dct: dict = dh.read_pkl('validity_domain', name)

    # Calculate grountruth
    ground_truth_data = np.zeros(len(score_dct['errors']))
    ground_truth_data[score_dct['errors'] > validity_domain_dct['error_threshold']] = 1

    # Calculate outlier share
    outlier_share = scoring.get_outlier_share(score_dct['errors'], validity_domain_dct['error_threshold'])

    # Save
    dh.write_pkl(ground_truth_data, 'ground_truth_data', name, override=False)
    dh.write_pkl(outlier_share, 'outlier_share', name, override=False)

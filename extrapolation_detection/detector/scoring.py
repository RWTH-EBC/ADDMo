import numpy as np
import sklearn.metrics as metrics
from numpy import ndarray


def score_samples(data_groundtruth: ndarray, data_nscores: ndarray, nscores_threshold: float, beta: float = None,
                  print_opt: bool = False, advanced: bool = False) -> dict:
    """Score novelty detection algorithm

    Parameters
    ----------
    data_groundtruth: ndarray
         Mx1 matrix, M: number of data points; Classification: 0: Normal data, 1: Outlier;
    data_nscores: ndarray
        Mx1 matrix of novelty scores, M: number of data points
    nscores_threshold: float
        Novelty detection threshold
    beta: float
        Beta value of F score
    print_opt: bool
        If true, scoring results are printed to console
    advanced: bool
        If true, advanced scoring metrics are calculated

    Returns
    -------
    dict
        Scoring results
    """
    # Classify data
    classification = np.zeros((len(data_nscores), 1))
    classification[data_nscores > nscores_threshold] = 1

    # True positive
    true_positive_mat = np.zeros((len(data_groundtruth), 1))
    true_positive_mat[((data_groundtruth == 1) & (classification == 1))] = 1
    true_positive = np.sum(true_positive_mat)

    # True negative
    true_negative_mat = np.zeros((len(data_groundtruth), 1))
    true_negative_mat[((data_groundtruth == 0) & (classification == 0))] = 1
    true_negative = np.sum(true_negative_mat)

    # False positive
    false_positive_mat = np.zeros((len(data_groundtruth), 1))
    false_positive_mat[((data_groundtruth == 0) & (classification == 1))] = 1
    false_positive = np.sum(false_positive_mat)

    # False negative
    false_negative_mat = np.zeros((len(data_groundtruth), 1))
    false_negative_mat[((data_groundtruth == 1) & (classification == 0))] = 1
    false_negative = np.sum(false_negative_mat)

    # Precision
    if (true_positive+false_positive) != 0:
        precision = true_positive / (true_positive+false_positive)
    else:
        precision = 1

    # Recall
    if (true_positive+false_negative) != 0:
        recall = true_positive / (true_positive+false_negative)
    else:
        recall = 1


    # Fscore
    if (precision+recall) == 0:
        f = 0
    else:
        f = 2*(precision*recall)/(precision+recall)

    # Scoring results
    result_dict = {'true_positive': true_positive, 'true_negative': true_negative, 'false_positive': false_positive,
                   'false_negative': false_negative, 'precision': precision, 'recall': recall, 'f': f}

    # Print option
    if print_opt:
        print('true_positive: ' + str(true_positive))
        print('true_negative: ' + str(true_negative))
        print('false_positive: ' + str(false_positive))
        print('false_negative: ' + str(false_negative))
        print('precision: ' + str(precision))
        print('recall: ' + str(recall))
        print('f: ' + str(f))

    # Fbeta score
    if beta is not None:
        if true_positive == 0:
            fbeta = 0
        else:
            fbeta = (1 + beta ** 2) * (precision * recall) / (beta ** 2 * precision + recall)
        result_dict['fbeta'] = fbeta
        if print_opt:
            print('fbeta: ' + str(fbeta))
    else:
        result_dict['fbeta'] = f

    # Advanced metrices
    if advanced:

        # ROC
        roc = round(metrics.roc_auc_score(data_groundtruth, data_nscores), ndigits=4)
        result_dict['roc'] = roc
        if print_opt:
            print('roc: ' + str(roc))

        roc_x, roc_y, _ = metrics.roc_curve(data_groundtruth, data_nscores)
        result_dict['roc_x'] = roc_x
        result_dict['roc_y'] = roc_y

        # PRC
        prc_y, prc_x, thresholds = metrics.precision_recall_curve(data_groundtruth, data_nscores)
        thresholds = np.append(thresholds, thresholds[-1])
        thresholds = (thresholds - min(thresholds))/(max(thresholds)-min(thresholds))
        result_dict['prc_x'] = prc_x
        result_dict['prc_y'] = prc_y
        result_dict['prc_thresholds'] = thresholds

        # Outlier share
        outliers = np.sum(data_groundtruth)
        outlier_share = outliers / len(data_groundtruth)

        # Relative f score compared to outlier share
        if beta is None:
            beta = 1
        rel_f = (f-(1+beta**2)*outlier_share/(1+beta**2 * outlier_share))/(1.0-(1+beta**2)*outlier_share /
                                                                           (1+beta**2 * outlier_share))
        result_dict['rel_f'] = rel_f

    return result_dict


def get_outlier_share(data_error, error_threshold):
    ground_truth = np.zeros((len(data_error), 1))
    ground_truth[data_error > error_threshold] = 1
    outliers = np.sum(ground_truth)
    return outliers / len(data_error)

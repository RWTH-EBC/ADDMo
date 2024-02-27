import numpy as np
import extrapolation_detection.machine_learning_util.data_handling as dh
from extrapolation_detection.detector import scoring


def score_clf(name: str, clf_name: str):
    """ Scores classifier with remaining data

    Parameters
    ----------
    name: str
        name of the ANN
    clf_name: str
        name of the classifier
    """
    data: dict = dh.read_pkl('data', name)
    clf = dh.read_pkl(clf_name, name)

    ns_scores = clf.score(data['non_available_data'].x_remaining)
    dh.write_pkl(ns_scores, 'ns_scores_' + clf_name, name, override=False)


def score_clf_ideal(name: str, clf_name: str):
    """ Scores (ideal) classifier with remaining data

    Parameters
    ----------
    name: str
        name of the ANN
    clf_name: str
        name of the classifier
    """
    data: dict = dh.read_pkl('data', name)
    clf = dh.read_pkl(clf_name + '_ideal', name)

    ns_scores = clf.score(data['non_available_data'].x_remaining)
    dh.write_pkl(ns_scores, 'ns_scores_' + clf_name + '_ideal', name, override=False)


def score_clf_2D(name: str, clf_name: str, contour_detail_error: int):
    """ Scores classifier for 2D plot

    Parameters
    ----------
    name: str
        name of the ANN
    clf_name: str
        name of the classifier
    contour_detail_error: int
        details of the decision boundary curve
    """

    # Load data
    data: dict = dh.read_pkl('data', name)
    clf = dh.read_pkl(clf_name, name)

    # Get bounds of 2D plot
    left = np.amin(np.concatenate((data['available_data'].xTrain, data['available_data'].xValid,
                                   data['available_data'].xTest, 
                                   data['non_available_data'].x_remaining))[:, 0])
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
    contour_error = np.zeros((contour_detail_error, contour_detail_error))
    # Evaluate meshgrid with classifier
    for i in range(0, contour_detail_error):
        for j in range(0, contour_detail_error):
            print(i)
            contour_error[i, j] = \
                clf.score(np.concatenate((np.array(xx[i, j]).reshape(-1, 1), np.array(yy[i, j]).reshape(-1, 1)),
                                         axis=1))
    score_2D_dct['contour_ns_scores'] = contour_error

    dh.write_pkl(score_2D_dct, 'errors_2D_' + clf_name, name, override=False)


def score_clf_ideal_2D(name: str, clf_name: str, contour_detail_error: int):
    """ Scores (ideal) classifier for 2D plot

    Parameters
    ----------
    name: str
        name of the ANN
    clf_name: str
        name of the classifier
    contour_detail_error: int
        details of the decision boundary curve
    """

    # Load data
    data: dict = dh.read_pkl('data', name)
    clf = dh.read_pkl(clf_name + '_ideal', name)

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
    contour_error = np.zeros((contour_detail_error, contour_detail_error))
    # Evaluate meshgrid with classifier
    for i in range(0, contour_detail_error):
        for j in range(0, contour_detail_error):
            contour_error[i, j] = \
                clf.score(np.concatenate((np.array(xx[i, j]).reshape(-1, 1), np.array(yy[i, j]).reshape(-1, 1)),
                                         axis=1))
    score_2D_dct['contour_ns_scores'] = contour_error

    dh.write_pkl(score_2D_dct, 'errors_2D_' + clf_name + '_ideal', name, override=False)


def evaluate(name: str, clf_name: str, beta: float = 1):
    """ Evaluate classifier with remaining data

    Parameters
    ----------
    name: str
        name of the ANN
    clf_name: str
        name of the classifier
    beta: float
        beta value for F-score
    """
    ns_scores: np.ndarray = dh.read_pkl('ns_scores_' + clf_name, name)
    ns_threshold: float = dh.read_pkl(clf_name + '_threshold', name)
    data_groundtruth: np.ndarray = dh.read_pkl('true_validity_classified_remaining', name)

    score = scoring.score_samples(data_groundtruth.reshape((-1, 1)), ns_scores, ns_threshold, beta=beta)

    dh.write_pkl(score, 'evaluation_' + clf_name, name, override=False)


def evaluate_ideal(name: str, clf_name: str, beta: float = 1):
    """ Evaluate (ideal) classifier with remaining data

    Parameters
    ----------
    name: str
        name of the ANN
    clf_name: str
        name of the classifier
    beta: float
        beta value for F-score
    """
    ns_scores: np.ndarray = dh.read_pkl('ns_scores_' + clf_name + '_ideal', name)
    ns_threshold: float = dh.read_pkl(clf_name + '_ideal' + '_threshold', name)
    data_groundtruth: np.ndarray = dh.read_pkl('true_validity_classified_remaining', name)

    score = scoring.score_samples(data_groundtruth.reshape((-1, 1)), ns_scores, ns_threshold, beta=beta)

    dh.write_pkl(score, 'evaluation_' + clf_name + '_ideal', name, override=False)

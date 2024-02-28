from abc import abstractmethod, ABC

import math

import numpy as np
from hyperopt import STATUS_OK, hp, fmin, tpe, Trials
from hyperopt.pyll import scope
from hyperopt.early_stop import no_progress_loss
from numpy import ndarray

from sklearn.gaussian_process.kernels import RBF

from extrapolation_detection.detector.detectors import D_KNN, D_OCSVM, D_ParzenWindow, D_GP, D_IF, \
    D_ABOD, D_LOF, D_MCD, D_GMM, \
    D_HBOS, D_ECOD, D_DSVDD, D_RNN, D_PCA, D_FB_KNN
from extrapolation_detection.detector.abstract_detector import AbstractDetector
from extrapolation_detection.detector.scoring import score_samples


class AbstractHyper(ABC):
    """Abstract class for hyperparameter optimization"""

    def __init__(self, outlier_fraction: float= 0.05, beta: float = 1, score_name: str = 'fbeta'):
        """

        Parameters
        ----------
        outlier_fraction: float
            PyOD specific: Expected fraction of outliers
        beta: float
            Beta value of F score
        score_name: str
            Name of score to be used for optimization
        """
        self.outlier_fraction: float = outlier_fraction
        self.beta: float = beta
        self.score_name: str = score_name

        self.hyper_threshold: Hyper_Threshold = Hyper_Threshold(beta=beta, score_name=score_name)

        self.x_train = None
        self.x_val = None
        self.groundtruth_val = None
        self.groundtruth_train = None

        self.clf = None

    def score(self) -> dict:
        """Score current classifier

        Returns
        -------
        dict
            Scoring results, e.g. 'loss', 'status', 'threshold'
        """
        # Train classifier
        self.clf.train(self.x_train)

        # Get novelty scores
        train_nscores = self.clf.get_decision_scores()
        val_nscores = self.clf.score(self.x_val)

        # Check, if training data should be included in validation
        if self.groundtruth_train is not None:
            val_nscores = np.concatenate((train_nscores, val_nscores))
            groundtruth = np.concatenate((self.groundtruth_train, self.groundtruth_val))
        else:
            groundtruth = self.groundtruth_val

        # Get novelty threshold through optimization
        nscores_threshold = self.hyper_threshold.opt(groundtruth, val_nscores)

        # Return score
        scoring = score_samples(groundtruth, val_nscores, nscores_threshold,
                                beta=self.beta, print_opt=False, advanced=False)
        return {'loss': -scoring[self.score_name], 'status': STATUS_OK, 'scoring': scoring,
                'threshold': nscores_threshold}

    @abstractmethod
    def opt(self) -> dict:
        """Optimize hyperparameters and return best result

        Returns
        -------
        dict
            returns best result
        """
        pass

    @abstractmethod
    def objective(self, space: dict) -> dict:
        """Creates and scores classifier from current search space

        Parameters
        ----------
        space: dict
            Current search space

        Returns
        -------
        dict
            Scoring results
        """
        pass

    @abstractmethod
    def get_clf(self, x_train: ndarray, x_val: ndarray, groundtruth_val: ndarray, groundtruth_train: ndarray = None) \
            -> tuple[AbstractDetector, float]:
        """Get classifier with optimized hyperparameters

        Parameters
        ----------
        x_train: ndarray
            NxD matrix of training data, N: number of data points, D: number of dimensions
        x_val: ndarray
            MxD matrix of validation data, M: number of data points, D: number of dimensions
        groundtruth_val: ndarray
             Mx1 matrix, M: number of validation data points; Classification: 0: Normal data, 1: Outlier;
        groundtruth_train: ndarray
            Optional Nx1 matrix, N: number of training data points; Classification: 0: Normal data, 1: Outlier;
            If not None, training data will be added to validation data for scoring

        Returns
        -------
         tuple[AbstractDetector, float]
            Classifier and novelty detection threshold
        """
        pass


class Hyper_KNN(AbstractHyper):
    """Hyperparameter optimization for k nearest neighbors"""

    def __init__(self, outlier_fraction: float, beta: float = 1, score_name: str = 'fbeta'):
        """

        Parameters
        ----------
        outlier_fraction: float
            PyOD specific: Expected fraction of outliers
        beta: float
            Beta value of F score
        score_name: str
            Name of score to be used for optimization
        """
        super(Hyper_KNN, self).__init__(outlier_fraction, beta, score_name)

    def opt(self) -> dict:
        """Optimize hyperparameters and return best result

        Returns
        -------
        dict
            returns best result
        """
        # Parameterize hyper optimization
        trials = Trials()
        best = fmin(self.objective,
                    space={'n_neighbors': scope.int(hp.quniform('n_neighbors', 2, 25, 1)),
                           'p': hp.choice('p', [1, 2]),
                           'method': hp.choice('method', ['largest', 'mean'])
                           },
                    algo=tpe.suggest,
                    max_evals=200,
                    early_stop_fn=no_progress_loss(75),
                    trials=trials)
        best['nscores_threshold'] = trials.best_trial['result']['threshold']
        return best

    def objective(self, space: dict) -> dict:
        """Creates and scores classifier from current search space

        Parameters
        ----------
        space: dict
            Current search space

        Returns
        -------
        dict
            Scoring results
        """
        n_neighbors = space['n_neighbors']
        p = space['p']
        method = space['method']
        # Create classifier
        self.clf: D_KNN = D_KNN(contamination=self.outlier_fraction, n_neighbors=n_neighbors, p=p, method=method)
        return self.score()

    def get_clf(self, x_train: ndarray, x_val: ndarray, groundtruth_val: ndarray, groundtruth_train: ndarray = None)\
            -> tuple[D_KNN, float]:
        """Get classifier with optimized hyperparameters

        Parameters
        ----------
        x_train: ndarray
            NxD matrix of training data, N: number of data points, D: number of dimensions
        x_val: ndarray
            MxD matrix of validation data, M: number of data points, D: number of dimensions
        groundtruth_val: ndarray
             Mx1 matrix, M: number of validation data points; Classification: 0: Normal data, 1: Outlier;
        groundtruth_train: ndarray
            Optional Nx1 matrix, N: number of training data points; Classification: 0: Normal data, 1: Outlier;
            If not None, training data will be added to validation data for scoring

        Returns
        -------
         tuple[AbstractDetector, float]
            Classifier and novelty detection threshold
        """
        # Data assignment
        self.x_train: ndarray = x_train
        self.x_val: ndarray = x_val
        self.groundtruth_val: ndarray = groundtruth_val
        self.groundtruth_train: ndarray = groundtruth_train

        # Optimization
        best = self.opt()

        # Create classifier
        knn_n_neighbors = math.trunc(best['n_neighbors'])
        print('KNN N_Neighbors : ' + str(knn_n_neighbors))
        p = [1, 2]
        knn_p = p[best['p']]
        print('KNN p: ' + str(knn_p))
        method = ['largest', 'mean']
        knn_method = method[best['method']]
        print('KNN method: ' + str(knn_method))
        nscores_threshold = best['nscores_threshold']
        clf = D_KNN(contamination=self.outlier_fraction, n_neighbors=knn_n_neighbors, p=knn_p, method=knn_method)
        clf.threshold = nscores_threshold
        return clf, nscores_threshold


class Hyper_FB_KNN(AbstractHyper):
    """Hyperparameter optimization for feature bagging using k nearest neighbors"""

    def __init__(self, outlier_fraction: float, beta: float = 1, score_name: str = 'fbeta'):
        """

        Parameters
        ----------
        outlier_fraction: float
            PyOD specific: Expected fraction of outliers
        beta: float
            Beta value of F score
        score_name: str
            Name of score to be used for optimization
        """
        super(Hyper_FB_KNN, self).__init__(outlier_fraction, beta, score_name)

    def opt(self) -> dict:
        """Optimize hyperparameters and return best result

        Returns
        -------
        dict
            returns best result
        """
        # Parameterize hyper optimization
        trials = Trials()
        best = fmin(self.objective,
                    space={'n_estimators': scope.int(hp.quniform('n_estimators', 1, 25, 1)),
                           'n_neighbors': scope.int(hp.quniform('n_neighbors', 2, 25, 1)),
                           'p': hp.choice('p', [1, 2]),
                           'method': hp.choice('method', ['largest', 'mean'])
                           },
                    algo=tpe.suggest,
                    max_evals=200,
                    early_stop_fn=no_progress_loss(100),
                    trials=trials)
        best['nscores_threshold'] = trials.best_trial['result']['threshold']
        return best

    def objective(self, space: dict) -> dict:
        """Creates and scores classifier from current search space

        Parameters
        ----------
        space: dict
            Current search space

        Returns
        -------
        dict
            Scoring results
        """
        n_estimators = space['n_estimators']
        n_neighbors = space['n_neighbors']
        p = space['p']
        method = space['method']
        # Create classifier
        self.clf: D_FB_KNN = D_FB_KNN(contamination=self.outlier_fraction, n_estimators=n_estimators,
                                      n_neighbors=n_neighbors, p=p, method=method)
        return self.score()

    def get_clf(self, x_train: ndarray, x_val: ndarray, groundtruth_val: ndarray, groundtruth_train: ndarray = None) \
            -> tuple[D_FB_KNN, float]:
        """Get classifier with optimized hyperparameters

        Parameters
        ----------
        x_train: ndarray
            NxD matrix of training data, N: number of data points, D: number of dimensions
        x_val: ndarray
            MxD matrix of validation data, M: number of data points, D: number of dimensions
        groundtruth_val: ndarray
             Mx1 matrix, M: number of validation data points; Classification: 0: Normal data, 1: Outlier;
        groundtruth_train: ndarray
            Optional Nx1 matrix, N: number of training data points; Classification: 0: Normal data, 1: Outlier;
            If not None, training data will be added to validation data for scoring

        Returns
        -------
         tuple[AbstractDetector, float]
            Classifier and novelty detection threshold
        """
        # Data assignment
        self.x_train: ndarray = x_train
        self.x_val: ndarray = x_val
        self.groundtruth_val: ndarray = groundtruth_val
        self.groundtruth_train: ndarray = groundtruth_train

        # Optimization
        best = self.opt()

        # Create classifier
        knn_n_estimators = math.trunc(best['n_estimators'])
        print('FB KNN N estimators: ' + str(knn_n_estimators))
        knn_n_neighbors = math.trunc(best['n_neighbors'])
        print('FB KNN N_Neighbors: ' + str(knn_n_neighbors))
        p = [1, 2]
        knn_p = p[best['p']]
        print('FB KNN p: ' + str(knn_p))
        method = ['largest', 'mean']
        knn_method = method[best['method']]
        print('FB KNN method: ' + str(knn_method))
        nscores_threshold = best['nscores_threshold']
        clf = D_FB_KNN(contamination=self.outlier_fraction, n_estimators=knn_n_estimators, n_neighbors=knn_n_neighbors,
                       p=knn_p, method=knn_method)
        clf.threshold = nscores_threshold
        return clf, nscores_threshold


class Hyper_LOF(AbstractHyper):
    """Hyperparameter optimization for k nearest neighbors"""

    def __init__(self, outlier_fraction: float, beta: float = 1, score_name: str = 'fbeta'):
        """

        Parameters
        ----------
        outlier_fraction: float
            PyOD specific: Expected fraction of outliers
        beta: float
            Beta value of F score
        score_name: str
            Name of score to be used for optimization
        """
        super(Hyper_LOF, self).__init__(outlier_fraction, beta, score_name)

    def opt(self) -> dict:
        """Optimize hyperparameters and return best result

        Returns
        -------
        dict
            returns best result
        """
        # Parameterize hyper optimization
        trials = Trials()
        best = fmin(self.objective,
                    space={'n_neighbors': scope.int(hp.quniform('n_neighbors', 2, 25, 1)),
                           'p': hp.choice('p', [1, 2]),
                           },
                    algo=tpe.suggest,
                    max_evals=150,
                    early_stop_fn=no_progress_loss(75),
                    trials=trials)
        best['nscores_threshold'] = trials.best_trial['result']['threshold']
        return best

    def objective(self, space: dict) -> dict:
        """Creates and scores classifier from current search space

        Parameters
        ----------
        space: dict
            Current search space

        Returns
        -------
        dict
            Scoring results
        """
        n_neighbors = space['n_neighbors']
        p = space['p']
        # Create classifier
        self.clf: D_LOF = D_LOF(contamination=self.outlier_fraction, n_neighbors=n_neighbors, p=p)
        return self.score()

    def get_clf(self, x_train: ndarray, x_val: ndarray, groundtruth_val: ndarray, groundtruth_train: ndarray = None)\
            -> tuple[D_LOF, float]:
        """Get classifier with optimized hyperparameters

        Parameters
        ----------
        x_train: ndarray
            NxD matrix of training data, N: number of data points, D: number of dimensions
        x_val: ndarray
            MxD matrix of validation data, M: number of data points, D: number of dimensions
        groundtruth_val: ndarray
             Mx1 matrix, M: number of validation data points; Classification: 0: Normal data, 1: Outlier;
        groundtruth_train: ndarray
            Optional Nx1 matrix, N: number of training data points; Classification: 0: Normal data, 1: Outlier;
            If not None, training data will be added to validation data for scoring

        Returns
        -------
         tuple[AbstractDetector, float]
            Classifier and novelty detection threshold
        """
        # Data assignment
        self.x_train: ndarray = x_train
        self.x_val: ndarray = x_val
        self.groundtruth_val: ndarray = groundtruth_val
        self.groundtruth_train: ndarray = groundtruth_train

        # Optimization
        best = self.opt()

        # Create classifier
        knn_n_neighbors = math.trunc(best['n_neighbors'])
        print('LOF N_Neighbors : ' + str(knn_n_neighbors))
        p = [1, 2]
        knn_p = p[best['p']]
        print('LOF p: ' + str(knn_p))
        nscores_threshold = best['nscores_threshold']
        clf = D_LOF(contamination=self.outlier_fraction, n_neighbors=knn_n_neighbors, p=knn_p)
        clf.threshold = nscores_threshold
        return clf, nscores_threshold


class Hyper_ABOD(AbstractHyper):
    """Hyperparameter optimization for angle based outlier detection"""

    def __init__(self, outlier_fraction: float, beta: float = 1, score_name: str = 'fbeta'):
        """

        Parameters
        ----------
        outlier_fraction: float
            PyOD specific: Expected fraction of outliers
        beta: float
            Beta value of F score
        score_name: str
            Name of score to be used for optimization
        """
        super(Hyper_ABOD, self).__init__(outlier_fraction, beta, score_name)

    def opt(self) -> dict:
        """Optimize hyperparameters and return best result

        Returns
        -------
        dict
            returns best result
        """
        # Parameterize hyper optimization
        trials = Trials()
        best = fmin(self.objective,
                    space={'n_neighbors': scope.int(hp.quniform('n_neighbors', 2, 25, 1)),
                           },
                    algo=tpe.suggest,
                    max_evals=50,
                    early_stop_fn=no_progress_loss(25),
                    trials=trials)
        best['nscores_threshold'] = trials.best_trial['result']['threshold']
        return best

    def objective(self, space: dict) -> dict:
        """Creates and scores classifier from current search space

        Parameters
        ----------
        space: dict
            Current search space

        Returns
        -------
        dict
            Scoring results
        """
        n_neighbors = space['n_neighbors']
        # Create classifier
        self.clf: D_ABOD = D_ABOD(contamination=self.outlier_fraction, n_neigbors=n_neighbors)
        return self.score()

    def get_clf(self, x_train: ndarray, x_val: ndarray, groundtruth_val: ndarray, groundtruth_train: ndarray = None) \
            -> tuple[D_ABOD, float]:
        """Get classifier with optimized hyperparameters

        Parameters
        ----------
        x_train: ndarray
            NxD matrix of training data, N: number of data points, D: number of dimensions
        x_val: ndarray
            MxD matrix of validation data, M: number of data points, D: number of dimensions
        groundtruth_val: ndarray
             Mx1 matrix, M: number of validation data points; Classification: 0: Normal data, 1: Outlier;
        groundtruth_train: ndarray
            Optional Nx1 matrix, N: number of training data points; Classification: 0: Normal data, 1: Outlier;
            If not None, training data will be added to validation data for scoring

        Returns
        -------
         tuple[AbstractDetector, float]
            Classifier and novelty detection threshold
        """
        # Data assignment
        self.x_train: ndarray = x_train
        self.x_val: ndarray = x_val
        self.groundtruth_val: ndarray = groundtruth_val
        self.groundtruth_train: ndarray = groundtruth_train

        # Optimization
        best = self.opt()

        # Create classifier
        n_neighbors = math.trunc(best['n_neighbors'])
        print('ABOD N_Neighbors : ' + str(n_neighbors))
        nscores_threshold = best['nscores_threshold']
        clf = D_ABOD(contamination=self.outlier_fraction, n_neigbors=n_neighbors)
        clf.threshold = nscores_threshold
        return clf, nscores_threshold


class Hyper_GMM(AbstractHyper):
    """Hyperparameter optimization for gaussian mixture model"""

    def __init__(self, outlier_fraction: float, beta: float = 1, score_name: str = 'fbeta'):
        """

        Parameters
        ----------
        outlier_fraction: float
            PyOD specific: Expected fraction of outliers
        beta: float
            Beta value of F score
        score_name: str
            Name of score to be used for optimization
        """
        super(Hyper_GMM, self).__init__(outlier_fraction, beta, score_name)

    def opt(self) -> dict:
        """Optimize hyperparameters and return best result

        Returns
        -------
        dict
            returns best result
        """
        # Parameterize hyper optimization
        trials = Trials()
        best = fmin(self.objective,
                    space={'n_components': scope.int(hp.quniform('n_components', 1, 50, 1)),
                           },
                    algo=tpe.suggest,
                    max_evals=50,
                    early_stop_fn=no_progress_loss(25),
                    trials=trials)
        best['nscores_threshold'] = trials.best_trial['result']['threshold']
        return best

    def objective(self, space: dict) -> dict:
        """Creates and scores classifier from current search space

        Parameters
        ----------
        space: dict
            Current search space

        Returns
        -------
        dict
            Scoring results
        """
        n_components = space['n_components']
        # Create classifier
        self.clf: D_GMM = D_GMM(contamination=self.outlier_fraction, n_components=n_components)
        return self.score()

    def get_clf(self, x_train: ndarray, x_val: ndarray, groundtruth_val: ndarray, groundtruth_train: ndarray = None) \
            -> tuple[D_GMM, float]:
        """Get classifier with optimized hyperparameters

        Parameters
        ----------
        x_train: ndarray
            NxD matrix of training data, N: number of data points, D: number of dimensions
        x_val: ndarray
            MxD matrix of validation data, M: number of data points, D: number of dimensions
        groundtruth_val: ndarray
             Mx1 matrix, M: number of validation data points; Classification: 0: Normal data, 1: Outlier;
        groundtruth_train: ndarray
            Optional Nx1 matrix, N: number of training data points; Classification: 0: Normal data, 1: Outlier;
            If not None, training data will be added to validation data for scoring

        Returns
        -------
         tuple[AbstractDetector, float]
            Classifier and novelty detection threshold
        """
        # Data assignment
        self.x_train: ndarray = x_train
        self.x_val: ndarray = x_val
        self.groundtruth_val: ndarray = groundtruth_val
        self.groundtruth_train: ndarray = groundtruth_train

        # Optimization
        best = self.opt()

        # Create classifier
        n_components = math.trunc(best['n_components'])
        print('GMM N_Neighbors : ' + str(n_components))
        nscores_threshold = best['nscores_threshold']
        clf = D_GMM(contamination=self.outlier_fraction, n_components=n_components)
        clf.threshold = nscores_threshold
        return clf, nscores_threshold


class Hyper_HBOS(AbstractHyper):
    """Hyperparameter optimization for histogram-based outlier detection"""

    def __init__(self, outlier_fraction: float, beta: float = 1, score_name: str = 'fbeta'):
        """

        Parameters
        ----------
        outlier_fraction: float
            PyOD specific: Expected fraction of outliers
        beta: float
            Beta value of F score
        score_name: str
            Name of score to be used for optimization
        """
        super(Hyper_HBOS, self).__init__(outlier_fraction, beta, score_name)

    def opt(self) -> dict:
        """Optimize hyperparameters and return best result

        Returns
        -------
        dict
            returns best result
        """
        # Parameterize hyper optimization
        trials = Trials()
        best = fmin(self.objective,
                    space={'n_bins': scope.int(hp.quniform('n_bins', 3, 50, 1)),
                           },
                    algo=tpe.suggest,
                    max_evals=50,
                    early_stop_fn=no_progress_loss(25),
                    trials=trials)
        best['nscores_threshold'] = trials.best_trial['result']['threshold']
        return best

    def objective(self, space: dict) -> dict:
        """Creates and scores classifier from current search space

        Parameters
        ----------
        space: dict
            Current search space

        Returns
        -------
        dict
            Scoring results
        """
        n_bins = space['n_bins']
        # Create classifier
        self.clf: D_HBOS = D_HBOS(contamination=self.outlier_fraction, n_bins=n_bins)
        return self.score()

    def get_clf(self, x_train: ndarray, x_val: ndarray, groundtruth_val: ndarray, groundtruth_train: ndarray = None) \
            -> tuple[D_HBOS, float]:
        """Get classifier with optimized hyperparameters

        Parameters
        ----------
        x_train: ndarray
            NxD matrix of training data, N: number of data points, D: number of dimensions
        x_val: ndarray
            MxD matrix of validation data, M: number of data points, D: number of dimensions
        groundtruth_val: ndarray
             Mx1 matrix, M: number of validation data points; Classification: 0: Normal data, 1: Outlier;
        groundtruth_train: ndarray
            Optional Nx1 matrix, N: number of training data points; Classification: 0: Normal data, 1: Outlier;
            If not None, training data will be added to validation data for scoring

        Returns
        -------
         tuple[AbstractDetector, float]
            Classifier and novelty detection threshold
        """
        # Data assignment
        self.x_train: ndarray = x_train
        self.x_val: ndarray = x_val
        self.groundtruth_val: ndarray = groundtruth_val
        self.groundtruth_train: ndarray = groundtruth_train

        # Optimization
        best = self.opt()

        # Create classifier
        n_bins = math.trunc(best['n_bins'])
        print('HBOS n bins: ' + str(n_bins))
        nscores_threshold = best['nscores_threshold']
        clf = D_HBOS(contamination=self.outlier_fraction, n_bins=n_bins)
        clf.threshold = nscores_threshold
        return clf, nscores_threshold


class Hyper_RNN(AbstractHyper):
    """Hyperparameter optimization for Auto Encoder / Replicator Neural Network"""

    def __init__(self, outlier_fraction: float, beta: float = 1, score_name: str = 'fbeta'):
        """

        Parameters
        ----------
        outlier_fraction: float
            PyOD specific: Expected fraction of outliers
        beta: float
            Beta value of F score
        score_name: str
            Name of score to be used for optimization
        """
        super(Hyper_RNN, self).__init__(outlier_fraction, beta, score_name)

    def opt(self) -> dict:
        """Optimize hyperparameters and return best result

        Returns
        -------
        dict
            returns best result
        """
        # Parameterize hyper optimization
        trials = Trials()
        best = fmin(self.objective,
                    space={'n_neurons': scope.int(hp.quniform('n_neurons', 2, self.x_train.shape[1] - 1, 2)),
                           },
                    algo=tpe.suggest,
                    max_evals=50,
                    early_stop_fn=no_progress_loss(25),
                    trials=trials)
        best['nscores_threshold'] = trials.best_trial['result']['threshold']
        return best

    def objective(self, space: dict) -> dict:
        """Creates and scores classifier from current search space

        Parameters
        ----------
        space: dict
            Current search space

        Returns
        -------
        dict
            Scoring results
        """
        n_neurons = space['n_neurons']
        # Create classifier
        self.clf: D_RNN = D_RNN(contamination=self.outlier_fraction, n_neurons=n_neurons)
        return self.score()

    def get_clf(self, x_train: ndarray, x_val: ndarray, groundtruth_val: ndarray, groundtruth_train: ndarray = None) \
            -> tuple[D_RNN, float]:
        """Get classifier with optimized hyperparameters

        Parameters
        ----------
        x_train: ndarray
            NxD matrix of training data, N: number of data points, D: number of dimensions
        x_val: ndarray
            MxD matrix of validation data, M: number of data points, D: number of dimensions
        groundtruth_val: ndarray
             Mx1 matrix, M: number of validation data points; Classification: 0: Normal data, 1: Outlier;
        groundtruth_train: ndarray
            Optional Nx1 matrix, N: number of training data points; Classification: 0: Normal data, 1: Outlier;
            If not None, training data will be added to validation data for scoring

        Returns
        -------
         tuple[AbstractDetector, float]
            Classifier and novelty detection threshold
        """
        # Data assignment
        self.x_train: ndarray = x_train
        self.x_val: ndarray = x_val
        self.groundtruth_val: ndarray = groundtruth_val
        self.groundtruth_train: ndarray = groundtruth_train

        # Optimization
        best = self.opt()

        # Create classifier
        n_neurons = math.trunc(best['n_neurons'])
        print('RNN N neurons: ' + str(n_neurons))
        nscores_threshold = best['nscores_threshold']
        clf = D_RNN(contamination=self.outlier_fraction, n_neurons=n_neurons)
        clf.threshold = nscores_threshold
        return clf, nscores_threshold


class Hyper_PCA(AbstractHyper):
    """Hyperparameter optimization for Principal Component Analysis"""

    def __init__(self, outlier_fraction: float, beta: float = 1, score_name: str = 'fbeta'):
        """

        Parameters
        ----------
        outlier_fraction: float
            PyOD specific: Expected fraction of outliers
        beta: float
            Beta value of F score
        score_name: str
            Name of score to be used for optimization
        """
        super(Hyper_PCA, self).__init__(outlier_fraction, beta, score_name)

    def opt(self) -> dict:
        """Optimize hyperparameters and return best result

        Returns
        -------
        dict
            returns best result
        """
        # Parameterize hyper optimization
        trials = Trials()
        best = fmin(self.objective,
                    space={'n_components': scope.int(hp.quniform('n_components', 1, self.x_train.shape[1], 2)),
                           },
                    algo=tpe.suggest,
                    max_evals=50,
                    early_stop_fn=no_progress_loss(25),
                    trials=trials)
        best['nscores_threshold'] = trials.best_trial['result']['threshold']
        return best

    def objective(self, space: dict) -> dict:
        """Creates and scores classifier from current search space

        Parameters
        ----------
        space: dict
            Current search space

        Returns
        -------
        dict
            Scoring results
        """
        n_components = space['n_components']
        # Create classifier
        self.clf: D_PCA = D_PCA(contamination=self.outlier_fraction, n_components=n_components)
        return self.score()

    def get_clf(self, x_train: ndarray, x_val: ndarray, groundtruth_val: ndarray, groundtruth_train: ndarray = None) \
            -> tuple[D_PCA, float]:
        """Get classifier with optimized hyperparameters

        Parameters
        ----------
        x_train: ndarray
            NxD matrix of training data, N: number of data points, D: number of dimensions
        x_val: ndarray
            MxD matrix of validation data, M: number of data points, D: number of dimensions
        groundtruth_val: ndarray
             Mx1 matrix, M: number of validation data points; Classification: 0: Normal data, 1: Outlier;
        groundtruth_train: ndarray
            Optional Nx1 matrix, N: number of training data points; Classification: 0: Normal data, 1: Outlier;
            If not None, training data will be added to validation data for scoring

        Returns
        -------
         tuple[AbstractDetector, float]
            Classifier and novelty detection threshold
        """
        # Data assignment
        self.x_train: ndarray = x_train
        self.x_val: ndarray = x_val
        self.groundtruth_val: ndarray = groundtruth_val
        self.groundtruth_train: ndarray = groundtruth_train

        # Optimization
        best = self.opt()

        # Create classifier
        n_components = math.trunc(best['n_components'])
        print('PCA N components: ' + str(n_components))
        nscores_threshold = best['nscores_threshold']
        clf = D_PCA(contamination=self.outlier_fraction, n_components=n_components)
        clf.threshold = nscores_threshold
        return clf, nscores_threshold


class Hyper_MCD(AbstractHyper):
    """Hyperparameter optimization for minimum covariance determinant"""

    def __init__(self, outlier_fraction: float, beta: float = 1, score_name: str = 'fbeta'):
        """

        Parameters
        ----------
        outlier_fraction: float
            PyOD specific: Expected fraction of outliers
        beta: float
            Beta value of F score
        score_name: str
            Name of score to be used for optimization
        """
        super(Hyper_MCD, self).__init__(outlier_fraction, beta, score_name)

    def opt(self) -> dict:
        """Optimize hyperparameters and return best result

        Returns
        -------
        dict
            returns best result
        """
        # Parameterize hyper optimization
        best = self.objective(dict())
        best['nscores_threshold'] = best['threshold']
        return best

    def objective(self, space: dict) -> dict:
        """Creates and scores classifier from current search space

        Parameters
        ----------
        space: dict
            Current search space

        Returns
        -------
        dict
            Scoring results
        """
        # Create classifier
        self.clf: D_MCD = D_MCD(contamination=self.outlier_fraction)
        return self.score()

    def get_clf(self, x_train: ndarray, x_val: ndarray, groundtruth_val: ndarray, groundtruth_train: ndarray = None)\
            -> tuple[D_MCD, float]:
        """Get classifier with optimized hyperparameters

        Parameters
        ----------
        x_train: ndarray
            NxD matrix of training data, N: number of data points, D: number of dimensions
        x_val: ndarray
            MxD matrix of validation data, M: number of data points, D: number of dimensions
        groundtruth_val: ndarray
             Mx1 matrix, M: number of validation data points; Classification: 0: Normal data, 1: Outlier;
        groundtruth_train: ndarray
            Optional Nx1 matrix, N: number of training data points; Classification: 0: Normal data, 1: Outlier;
            If not None, training data will be added to validation data for scoring

        Returns
        -------
         tuple[AbstractDetector, float]
            Classifier and novelty detection threshold
        """
        # Data assignment
        self.x_train: ndarray = x_train
        self.x_val: ndarray = x_val
        self.groundtruth_val: ndarray = groundtruth_val
        self.groundtruth_train: ndarray = groundtruth_train

        # Optimization
        best = self.opt()

        # Create classifier
        nscores_threshold = best['nscores_threshold']
        clf = D_MCD(contamination=self.outlier_fraction)
        clf.threshold = nscores_threshold
        return clf, nscores_threshold


class Hyper_ECOD(AbstractHyper):
    """Hyperparameter optimization for Empirical Cumulative Distribution Functions (ECOD)"""

    def __init__(self, outlier_fraction: float, beta: float = 1, score_name: str = 'fbeta'):
        """

        Parameters
        ----------
        outlier_fraction: float
            PyOD specific: Expected fraction of outliers
        beta: float
            Beta value of F score
        score_name: str
            Name of score to be used for optimization
        """
        super(Hyper_ECOD, self).__init__(outlier_fraction, beta, score_name)

    def opt(self) -> dict:
        """Optimize hyperparameters and return best result

        Returns
        -------
        dict
            returns best result
        """
        # Parameterize hyper optimization
        best = self.objective(dict())
        best['nscores_threshold'] = best['threshold']
        return best

    def objective(self, space: dict) -> dict:
        """Creates and scores classifier from current search space

        Parameters
        ----------
        space: dict
            Current search space

        Returns
        -------
        dict
            Scoring results
        """
        # Create classifier
        self.clf: D_ECOD = D_ECOD(contamination=self.outlier_fraction)
        return self.score()

    def get_clf(self, x_train: ndarray, x_val: ndarray, groundtruth_val: ndarray, groundtruth_train: ndarray = None) \
            -> tuple[D_ECOD, float]:
        """Get classifier with optimized hyperparameters

        Parameters
        ----------
        x_train: ndarray
            NxD matrix of training data, N: number of data points, D: number of dimensions
        x_val: ndarray
            MxD matrix of validation data, M: number of data points, D: number of dimensions
        groundtruth_val: ndarray
             Mx1 matrix, M: number of validation data points; Classification: 0: Normal data, 1: Outlier;
        groundtruth_train: ndarray
            Optional Nx1 matrix, N: number of training data points; Classification: 0: Normal data, 1: Outlier;
            If not None, training data will be added to validation data for scoring

        Returns
        -------
         tuple[AbstractDetector, float]
            Classifier and novelty detection threshold
        """
        # Data assignment
        self.x_train: ndarray = x_train
        self.x_val: ndarray = x_val
        self.groundtruth_val: ndarray = groundtruth_val
        self.groundtruth_train: ndarray = groundtruth_train

        # Optimization
        best = self.opt()

        # Create classifier
        nscores_threshold = best['nscores_threshold']
        clf = D_ECOD(contamination=self.outlier_fraction)
        clf.threshold = nscores_threshold
        return clf, nscores_threshold


class Hyper_DSVDD(AbstractHyper):
    """Hyperparameter optimization for Deep One-Class Classification"""

    def __init__(self, outlier_fraction: float, beta: float = 1, score_name: str = 'fbeta'):
        """

        Parameters
        ----------
        outlier_fraction: float
            PyOD specific: Expected fraction of outliers
        beta: float
            Beta value of F score
        score_name: str
            Name of score to be used for optimization
        """
        super(Hyper_DSVDD, self).__init__(outlier_fraction, beta, score_name)

    def opt(self) -> dict:
        """Optimize hyperparameters and return best result

        Returns
        -------
        dict
            returns best result
        """
        # Parameterize hyper optimization
        trials = Trials()
        best = fmin(self.objective,
                    space={'n_neurons': scope.int(hp.quniform('n_neurons', 2, self.x_train.shape[1] - 1, 2)),
                           },
                    algo=tpe.suggest,
                    max_evals=50,
                    early_stop_fn=no_progress_loss(25),
                    trials=trials)
        best['nscores_threshold'] = trials.best_trial['result']['threshold']
        return best

    def objective(self, space: dict) -> dict:
        """Creates and scores classifier from current search space

        Parameters
        ----------
        space: dict
            Current search space

        Returns
        -------
        dict
            Scoring results
        """
        n_neurons = space['n_neurons']
        # Create classifier
        self.clf: D_DSVDD = D_DSVDD(contamination=self.outlier_fraction, n_neurons=n_neurons)
        return self.score()

    def get_clf(self, x_train: ndarray, x_val: ndarray, groundtruth_val: ndarray, groundtruth_train: ndarray = None)\
            -> tuple[D_DSVDD, float]:
        """Get classifier with optimized hyperparameters

        Parameters
        ----------
        x_train: ndarray
            NxD matrix of training data, N: number of data points, D: number of dimensions
        x_val: ndarray
            MxD matrix of validation data, M: number of data points, D: number of dimensions
        groundtruth_val: ndarray
             Mx1 matrix, M: number of validation data points; Classification: 0: Normal data, 1: Outlier;
        groundtruth_train: ndarray
            Optional Nx1 matrix, N: number of training data points; Classification: 0: Normal data, 1: Outlier;
            If not None, training data will be added to validation data for scoring

        Returns
        -------
         tuple[AbstractDetector, float]
            Classifier and novelty detection threshold
        """
        # Data assignment
        self.x_train: ndarray = x_train
        self.x_val: ndarray = x_val
        self.groundtruth_val: ndarray = groundtruth_val
        self.groundtruth_train: ndarray = groundtruth_train

        # Optimization
        best = self.opt()

        n_neurons = math.trunc(best['n_neurons'])
        print('RNN N neurons: ' + str(n_neurons))
        nscores_threshold = best['nscores_threshold']
        # Create classifier
        clf = D_DSVDD(contamination=self.outlier_fraction, n_neurons=n_neurons)
        clf.threshold = nscores_threshold
        return clf, nscores_threshold


class Hyper_OCSVM(AbstractHyper):
    """Hyperparameter optimization for one class support vector machine"""

    def __init__(self, outlier_fraction: float, beta: float = 1, score_name: str = 'fbeta'):
        """

        Parameters
        ----------
        outlier_fraction: float
            PyOD specific: Expected fraction of outliers
        beta: float
            Beta value of F score
        score_name: str
            Name of score to be used for optimization
        """
        super(Hyper_OCSVM, self).__init__(outlier_fraction, beta, score_name)

    def opt(self) -> dict:
        """Optimize hyperparameters and return best result

        Returns
        -------
        dict
            returns best result
        """
        # Parameterize hyper optimization
        trials = Trials()
        best = fmin(self.objective,
                    space={'gamma': hp.uniform('gamma', 0.1, 150),
                           # 'nu': hp.uniform('nu', 0.001, 0.99)
                           },
                    algo=tpe.suggest,
                    max_evals=120,
                    early_stop_fn=no_progress_loss(60),
                    trials=trials)
        best['nscores_threshold'] = trials.best_trial['result']['threshold']
        best['nu'] = self.outlier_fraction
        return best

    def objective(self, space: dict) -> dict:
        """Creates and scores classifier from current search space

        Parameters
        ----------
        space: dict
            Current search space

        Returns
        -------
        dict
            Scoring results
        """
        # nu = space['nu']
        nu = self.outlier_fraction
        gamma = space['gamma']
        # Create classifier
        self.clf: D_OCSVM = D_OCSVM(contamination=self.outlier_fraction, nu=nu, gamma=gamma)
        return self.score()

    def get_clf(self, x_train: ndarray, x_val: ndarray, groundtruth_val: ndarray, groundtruth_train: ndarray = None) \
            -> tuple[D_OCSVM, float]:
        """Get classifier with optimized hyperparameters

        Parameters
        ----------
        x_train: ndarray
            NxD matrix of training data, N: number of data points, D: number of dimensions
        x_val: ndarray
            MxD matrix of validation data, M: number of data points, D: number of dimensions
        groundtruth_val: ndarray
             Mx1 matrix, M: number of validation data points; Classification: 0: Normal data, 1: Outlier;
        groundtruth_train: ndarray
            Optional Nx1 matrix, N: number of training data points; Classification: 0: Normal data, 1: Outlier;
            If not None, training data will be added to validation data for scoring

        Returns
        -------
         tuple[AbstractDetector, float]
            Classifier and novelty detection threshold
        """
        # Data assignment
        self.x_train: ndarray = x_train
        self.x_val: ndarray = x_val
        self.groundtruth_val: ndarray = groundtruth_val
        self.groundtruth_train: ndarray = groundtruth_train

        # Optimization
        best = self.opt()

        # Create classifier
        ocsvm_nu = best['nu']
        ocsvm_gamma = best['gamma']
        print('OCSVM Gamma : ' + str(ocsvm_gamma))
        nscores_threshold = best['nscores_threshold']
        clf = D_OCSVM(contamination=self.outlier_fraction, nu=ocsvm_nu, gamma=ocsvm_gamma)
        clf.threshold = nscores_threshold
        return clf, nscores_threshold


class Hyper_KDE(AbstractHyper):
    """Hyperparameter optimization for kernel density estimation"""

    def __init__(self, outlier_fraction: float, beta: float = 1, score_name: str = 'fbeta'):
        """

        Parameters
        ----------
        outlier_fraction: float
            PyOD specific: Expected fraction of outliers
        beta: float
            Beta value of F score
        score_name: str
            Name of score to be used for optimization
        """
        super(Hyper_KDE, self).__init__(outlier_fraction, beta, score_name)

    def opt(self) -> dict:
        """Optimize hyperparameters and return best result

        Returns
        -------
        dict
            returns best result
        """
        # Parameterize hyper optimization
        trials = Trials()
        best = fmin(self.objective,
                    space={'bandwith': hp.uniform('bandwith', 0.001, 1.5)},
                    algo=tpe.suggest,
                    max_evals=120,
                    early_stop_fn=no_progress_loss(60),
                    trials=trials)
        best['nscores_threshold'] = trials.best_trial['result']['threshold']
        return best

    def objective(self, space: dict) -> dict:
        """Creates and scores classifier from current search space

        Parameters
        ----------
        space: dict
            Current search space

        Returns
        -------
        dict
            Scoring results
        """
        bandwith = space['bandwith']
        # Create classifier
        self.clf: D_ParzenWindow = D_ParzenWindow(contamination=self.outlier_fraction, bandwith=bandwith)
        return self.score()

    def get_clf(self, x_train: ndarray, x_val: ndarray, groundtruth_val: ndarray, groundtruth_train: ndarray = None)\
            -> tuple[D_ParzenWindow, float]:
        """Get classifier with optimized hyperparameters

        Parameters
        ----------
        x_train: ndarray
            NxD matrix of training data, N: number of data points, D: number of dimensions
        x_val: ndarray
            MxD matrix of validation data, M: number of data points, D: number of dimensions
        groundtruth_val: ndarray
             Mx1 matrix, M: number of validation data points; Classification: 0: Normal data, 1: Outlier;
        groundtruth_train: ndarray
            Optional Nx1 matrix, N: number of training data points; Classification: 0: Normal data, 1: Outlier;
            If not None, training data will be added to validation data for scoring

        Returns
        -------
         tuple[AbstractDetector, float]
            Classifier and novelty detection threshold
        """
        # Data assignment
        self.x_train: ndarray = x_train
        self.x_val: ndarray = x_val
        self.groundtruth_val: ndarray = groundtruth_val
        self.groundtruth_train: ndarray = groundtruth_train

        # Optimization
        best = self.opt()

        # Create classifier
        kde_bandwith = best['bandwith']
        print('KDE Bandwith : ' + str(kde_bandwith))
        nscores_threshold = best['nscores_threshold']
        clf = D_ParzenWindow(contamination=self.outlier_fraction, bandwith=kde_bandwith)
        clf.threshold = nscores_threshold
        return clf, nscores_threshold


class Hyper_GP(AbstractHyper):
    """Hyperparameter optimization for gaussian process regression"""

    def __init__(self, outlier_fraction: float, beta: float = 1, score_name: str = 'fbeta'):
        """

        Parameters
        ----------
        outlier_fraction: float
            PyOD specific: Expected fraction of outliers
        beta: float
            Beta value of F score
        score_name: str
            Name of score to be used for optimization
        """
        super(Hyper_GP, self).__init__(outlier_fraction, beta, score_name)

    def opt(self) -> dict:
        """Optimize hyperparameters and return best result

        Returns
        -------
        dict
            returns best result
        """
        # Parameterize hyper optimization
        trials = Trials()
        best = fmin(self.objective,
                    space={'length_scale': hp.uniform('length_scale', 0.01, 1100)},
                    algo=tpe.suggest,
                    max_evals=120,
                    early_stop_fn=no_progress_loss(60),
                    trials=trials)
        best['nscores_threshold'] = trials.best_trial['result']['threshold']
        return best

    def objective(self, space: dict) -> dict:
        """Creates and scores classifier from current search space

        Parameters
        ----------
        space: dict
            Current search space

        Returns
        -------
        dict
            Scoring results
        """
        length_scale = space['length_scale']
        # Create classifier
        self.clf: D_GP = D_GP(contamination=self.outlier_fraction,
                              kernel=1.0 * RBF(length_scale=length_scale, length_scale_bounds='fixed'))
        return self.score()

    def get_clf(self, x_train: ndarray, x_val: ndarray, groundtruth_val: ndarray, groundtruth_train: ndarray = None) \
            -> tuple[D_GP, float]:
        """Get classifier with optimized hyperparameters

        Parameters
        ----------
        x_train: ndarray
            NxD matrix of training data, N: number of data points, D: number of dimensions
        x_val: ndarray
            MxD matrix of validation data, M: number of data points, D: number of dimensions
        groundtruth_val: ndarray
             Mx1 matrix, M: number of validation data points; Classification: 0: Normal data, 1: Outlier;
        groundtruth_train: ndarray
            Optional Nx1 matrix, N: number of training data points; Classification: 0: Normal data, 1: Outlier;
            If not None, training data will be added to validation data for scoring

        Returns
        -------
         tuple[AbstractDetector, float]
            Classifier and novelty detection threshold
        """
        # Data assignment
        self.x_train: ndarray = x_train
        self.x_val: ndarray = x_val
        self.groundtruth_val: ndarray = groundtruth_val
        self.groundtruth_train: ndarray = groundtruth_train

        # Optimization
        best = self.opt()

        # Create classifier
        gp_length_scale = best['length_scale']
        print('GP Length_Scale : ' + str(gp_length_scale))
        nscores_threshold = best['nscores_threshold']
        clf = D_GP(contamination=self.outlier_fraction,
                   kernel=1.0 * RBF(length_scale=gp_length_scale, length_scale_bounds='fixed'))
        clf.threshold = nscores_threshold
        return clf, nscores_threshold


class Hyper_IF(AbstractHyper):
    """Hyperparameter optimization for isolation forest"""

    def __init__(self, outlier_fraction: float, beta: float = 1, score_name: str = 'fbeta'):
        """

        Parameters
        ----------
        outlier_fraction: float
            PyOD specific: Expected fraction of outliers
        beta: float
            Beta value of F score
        score_name: str
            Name of score to be used for optimization
        """
        super(Hyper_IF, self).__init__(outlier_fraction, beta, score_name)

    def opt(self) -> dict:
        """Optimize hyperparameters and return best result

        Returns
        -------
        dict
            returns best result
        """
        # Parameterize hyper optimization
        trials = Trials()
        best = fmin(self.objective,
                    space={'seed': hp.randint('seed', 10000)},
                    algo=tpe.suggest,
                    max_evals=120,
                    early_stop_fn=no_progress_loss(60),
                    trials=trials)
        best['nscores_threshold'] = trials.best_trial['result']['threshold']
        return best

    def objective(self, space: dict) -> dict:
        """Creates and scores classifier from current search space

        Parameters
        ----------
        space: dict
            Current search space

        Returns
        -------
        dict
            Scoring results
        """
        seed = space['seed']
        # Create classifier
        self.clf: D_IF = D_IF(contamination=self.outlier_fraction, random_state=seed)
        return self.score()

    def get_clf(self, x_train: ndarray, x_val: ndarray, groundtruth_val: ndarray, groundtruth_train: ndarray = None) \
            -> tuple[D_IF, float]:
        """Get classifier with optimized hyperparameters

        Parameters
        ----------
        x_train: ndarray
            NxD matrix of training data, N: number of data points, D: number of dimensions
        x_val: ndarray
            MxD matrix of validation data, M: number of data points, D: number of dimensions
        groundtruth_val: ndarray
             Mx1 matrix, M: number of validation data points; Classification: 0: Normal data, 1: Outlier;
        groundtruth_train: ndarray
            Optional Nx1 matrix, N: number of training data points; Classification: 0: Normal data, 1: Outlier;
            If not None, training data will be added to validation data for scoring

        Returns
        -------
         tuple[AbstractDetector, float]
            Classifier and novelty detection threshold
        """
        # Data assignment
        self.x_train: ndarray = x_train
        self.x_val: ndarray = x_val
        self.groundtruth_val: ndarray = groundtruth_val
        self.groundtruth_train: ndarray = groundtruth_train

        # Optimization
        best = self.opt()

        # Create classifier
        seed = best['seed']
        nscores_threshold = best['nscores_threshold']
        print('IF seed : ' + str(seed))
        clf = D_IF(contamination=self.outlier_fraction, random_state=seed)
        clf.threshold = nscores_threshold
        return clf, nscores_threshold


class Hyper_Threshold:
    """Hyperparameter optimization for novelty detection threshold"""

    def __init__(self, beta: float = 1, score_name: str = 'fbeta'):
        """

        Parameters
        ----------
        beta: float
            Beta value of F score
        score_name: str
            Name of score to be used for optimization
        """
        self.beta: float = beta
        self.score_name: str = score_name

        self.groundtruth = None
        self.nscores = None
        self.nscores_threshold = None

    def score(self) -> dict:
        """Score current threshold

        Returns
        -------
        dict
            Scoring results, e.g. 'loss', 'status'
        """
        scoring = score_samples(self.groundtruth, self.nscores, self.nscores_threshold,
                                beta=self.beta, print_opt=False, advanced=False)
        return {'loss': -scoring[self.score_name], 'status': STATUS_OK, 'scoring': scoring}

    def opt(self, groundtruth: ndarray, nscores: ndarray) -> float:
        """Optimize novelty detection threshold
        self.groundtruth = groundtruth
        self.nscores = nscores
        if min(self.nscores) < max(self.nscores):
            best = fmin(self.objective,
                        space={'threshold': hp.uniform('threshold', min(self.nscores), max(self.nscores))},
                        algo=tpe.suggest,
                        max_evals=50,
                        early_stop_fn=no_progress_loss(25),
                        verbose=True)
            return best['threshold']
        else:
            return min(self.nscores)

        Parameters
        ----------
        groundtruth: ndarray
             Mx1 matrix, M: number of data points; Classification: 0: Normal data, 1: Outlier;
        nscores: ndarray
            Mx1 matrix of novelty scores, M: number of data points

        Returns
        -------
        float
            Optimized novelty detection threshold
        """
        # Data assignment
        self.groundtruth: ndarray = groundtruth
        self.nscores: ndarray = nscores

        # Optimization
        if min(self.nscores) < max(self.nscores):
            best = fmin(self.objective,
                        space={'threshold': hp.uniform('threshold', min(self.nscores), max(self.nscores))},
                        algo=tpe.suggest,
                        max_evals=50,
                        early_stop_fn=no_progress_loss(25),
                        verbose=True)
            return best['threshold']
        else:
            return min(self.nscores)

    def objective(self, space: dict) -> dict:
        """Creates and scores classifier from current search space

        Parameters
        ----------
        space: dict
            Current search space

        Returns
        -------
        dict
            Scoring results
        """
        self.nscores_threshold: float = space['threshold']
        return self.score()

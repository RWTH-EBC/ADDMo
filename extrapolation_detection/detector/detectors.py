from abc import ABC

import numpy as np
from numpy import ndarray

from pyod.models.abod import ABOD
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.deep_svdd import DeepSVDD
from pyod.models.ecod import ECOD
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.models.iforest import IForest
from pyod.models.mcd import MCD
from pyod.models.knn import KNN
from pyod.models.pca import PCA

from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings

from extrapolation_detection.detector.abstract_detector import AbstractDetector
class D_OCSVM(AbstractDetector):
    """One class support vector machine"""

    def __init__(self, contamination: float = 0.01, nu: float = 0.15, kernel: str = 'rbf', gamma: float or str = 10):
        """

        Parameters
        ----------
        contamination: float
            PyOD specific: Expected fraction of outliers
        nu: float
            Nu parameter
        kernel: str
            Kernel parameter
        gamma: float or str
            Gamma parameter
        """
        super(D_OCSVM).__init__()
        self.nu: float = nu
        self.kernel: str = kernel
        self.gamma: float = gamma
        self.contamination: float = contamination
        self.clf: OCSVM = OCSVM(nu=self.nu, kernel=self.kernel, gamma=self.gamma, contamination=contamination)

    @property
    def info(self) -> dict:
        """Additional classifier information

        Returns
        -------
        dict
        """
        return {'Gamma': self.gamma, 'Nu': self.nu, 'Kernel': self.kernel, 'NoveltyThreshold': self.threshold}


class D_IF(AbstractDetector):
    """Isolation Forest"""

    def __init__(self, contamination: float = 0.01, random_state: float = None):
        """

        Parameters
        ----------
        contamination: float
            PyOD specific: Expected fraction of outliers
        random_state: float
            Seed of random state
        """
        super(D_IF).__init__()
        self.contamination: float = contamination
        if random_state is not None:
            self.random_state: float = random_state
            self.clf: IForest = IForest(contamination=contamination, random_state=random_state)
        else:
            self.clf: IForest = IForest(contamination=contamination)

    @property
    def info(self) -> dict:
        """Additional classifier information

        Returns
        -------
        dict
        """
        return {'Seed': self.clf.random_state, 'NoveltyThreshold': self.threshold}


class D_MCD(AbstractDetector):
    """Minimum Covariance Determinant"""

    def __init__(self, contamination: float = 0.01):
        """

        Parameters
        ----------
        contamination: float
             PyOD specific: Expected fraction of outliers
        """
        super(D_MCD).__init__()
        self.contamination: float = contamination
        self.clf: MCD = MCD(contamination=contamination)

    @property
    def info(self) -> dict:
        """Additional classifier information

        Returns
        -------
        dict
        """
        return {'NoveltyThreshold': self.threshold}


class D_ECOD(AbstractDetector):
    """Empirical Cumulative Distribution Functions (ECOD)"""

    def __init__(self, contamination: float = 0.01):
        """

        Parameters
        ----------
        contamination: float
             PyOD specific: Expected fraction of outliers
        """
        super(D_ECOD).__init__()
        self.contamination: float = contamination
        self.clf: ECOD = ECOD(contamination=contamination)

    @property
    def info(self) -> dict:
        """Additional classifier information

        Returns
        -------
        dict
        """
        return {'NoveltyThreshold': self.threshold}


class D_DSVDD(AbstractDetector):
    """Deep One-Class Classification for outlier detection"""

    def __init__(self, contamination: float = 0.01, n_neurons: int = 32):
        """

        Parameters
        ----------
        contamination: float
             PyOD specific: Expected fraction of outliers
        """
        super(D_DSVDD).__init__()
        self.n_neurons = n_neurons
        self.contamination: float = contamination
        self.clf: DeepSVDD = DeepSVDD(contamination=contamination, hidden_neurons=[n_neurons*2, n_neurons])

    @property
    def info(self) -> dict:
        """Additional classifier information

        Returns
        -------
        dict
        """
        return {'NoveltyThreshold': self.threshold, 'N_neurons': self.n_neurons}


class D_KNN(AbstractDetector):
    """K nearest neighbors"""

    def __init__(self, contamination: float = 0.01, n_neighbors: int = 5, method: str = 'largest', p: int = 2):
        """

        Parameters
        ----------
        contamination: float
            PyOD specific: Expected fraction of outliers
        n_neighbors: int
            k parameter
        method: str
            Distance method, e.g. 'largest', 'mean'
        p: int
            Distance function, p=1: Manhattan, p=2: Euclidean
        """
        super(D_KNN).__init__()
        self.contamination: float = contamination
        self.n_neighbors: int = n_neighbors
        self.method: str = method
        self.p: int = p
        self.clf: KNN = KNN(contamination=contamination, n_neighbors=n_neighbors, method=method, p=p)

    @property
    def info(self) -> dict:
        """Additional classifier information

        Returns
        -------
        dict
        """
        return {'K': self.n_neighbors, 'p': self.p, 'Method': self.method, 'NoveltyThreshold': self.threshold}


class D_FB_KNN(AbstractDetector):
    """Feature bagging with K nearest neighbors"""

    def __init__(self, contamination: float = 0.01, n_estimators: int = 10, n_neighbors: int = 5,
                 method: str = 'largest', p: int = 2):
        """

        Parameters
        ----------
        contamination: float
            PyOD specific: Expected fraction of outliers
        n_neighbors: int
            k parameter
        method: str
            Distance method, e.g. 'largest', 'mean'
        p: int
            Distance function, p=1: Manhattan, p=2: Euclidean
        """
        super(D_FB_KNN).__init__()
        self.contamination: float = contamination
        self.n_estimators = n_estimators
        self.n_neighbors: int = n_neighbors
        self.method: str = method
        self.p: int = p
        self.clf: FeatureBagging = FeatureBagging(contamination=contamination, random_state=1000,
                                                  n_estimators=n_estimators,
                                                  base_estimator=KNN(contamination=contamination,
                                                                     n_neighbors=n_neighbors,
                                                                     method=method, p=p))

    @property
    def info(self) -> dict:
        """Additional classifier information

        Returns
        -------
        dict
        """
        return {'N estimators': self.n_estimators, 'K': self.n_neighbors, 'p': self.p, 'Method': self.method,
                'NoveltyThreshold': self.threshold}


class D_ABOD(AbstractDetector):
    """Angle based outlier detection"""

    def __init__(self, contamination: float = 0.01, n_neigbors: int = 5):
        """

        Parameters
        ----------
        contamination: float
            PyOD specific: Expected fraction of outliers
        n_neigbors: int
            Number neighbors parameter
        """
        super(D_ABOD).__init__()
        self.contamination: float = contamination
        self.n_neigbors: int = n_neigbors
        self.clf: ABOD = ABOD(contamination=contamination, n_neighbors=n_neigbors)

    def score(self, x_test: ndarray) -> ndarray:
        """Get novelty scores for training data

        Parameters
        ----------
        x_test: ndarray
            Test data to be scored: NxD matrix, N: number of data points, D: number of dimensions
        Returns
        -------
        ndarray
            Novelty scores;  Nx1 matrix, N: number of data points
        """
        scores = super().score(x_test)
        return -np.log(scores * -1)

    def get_decision_scores(self) -> ndarray:
        """Get novelty scores of training data

        Returns
        -------
        ndarray
            Novelty scores of training data; Nx1 matrix, N: number of data points
        """
        return -np.log(self.clf.decision_scores_ * -1)

    @property
    def info(self) -> dict:
        """Additional classifier information

        Returns
        -------
        dict
        """
        return {'N-Neigbors': self.n_neigbors, 'NoveltyThreshold': self.threshold}


class D_HBOS(AbstractDetector):
    """Histogram-based Outlier Detection"""

    def __init__(self, contamination: float = 0.01, n_bins: int = 10):
        """

        Parameters
        ----------
        contamination: float
            PyOD specific: Expected fraction of outliers
        n_bins: int
            Number of bins parameter
        """
        super(D_HBOS).__init__()
        self.contamination: float = contamination
        self.n_bins: int = n_bins
        self.clf: HBOS = HBOS(contamination=contamination, n_bins=n_bins)

    @property
    def info(self) -> dict:
        """Additional classifier information

        Returns
        -------
        dict
        """
        return {'N Bins': self.n_bins, 'NoveltyThreshold': self.threshold}


class D_RNN(AbstractDetector):
    """Auto Encoder / Replicator Neural Network"""

    def __init__(self, contamination: float = 0.01, n_neurons: int = 32):
        """

        Parameters
        ----------
        contamination: float
            PyOD specific: Expected fraction of outliers
        n_neurons: int
            Number of neurons per hidden layer parameter [N*2, N, N, N*2]
        """
        super(D_RNN).__init__()
        self.contamination: float = contamination
        self.n_neurons: int = n_neurons
        self.hidden_layer = [n_neurons*2, n_neurons, n_neurons, n_neurons*2]
        self.clf: AutoEncoder = AutoEncoder(contamination=contamination, hidden_neurons=self.hidden_layer)

    @property
    def info(self) -> dict:
        """Additional classifier information

        Returns
        -------
        dict
        """
        return {'Hidden layer': self.hidden_layer, 'NoveltyThreshold': self.threshold}


class D_PCA(AbstractDetector):
    """Principal Component Analysis"""

    def __init__(self, contamination: float = 0.01, n_components: int = None):
        """

        Parameters
        ----------
        contamination: float
            PyOD specific: Expected fraction of outliers
        n_components: int
            Number of principal components parameter
        """
        super(D_PCA).__init__()
        self.contamination: float = contamination
        self.n_components: int = n_components
        self.clf: PCA = PCA(contamination=contamination, n_components=self.n_components)

    @property
    def info(self) -> dict:
        """Additional classifier information

        Returns
        -------
        dict
        """
        return {'N components': self.n_components, 'NoveltyThreshold': self.threshold}


class D_LOF(AbstractDetector):
    """Local outlier factor"""

    def __init__(self, contamination: float = 0.01, n_neighbors: int = 20, p: int = 2):
        """

        Parameters
        ----------
        contamination: float
            PyOD specific: Expected fraction of outliers
        n_neighbors: int
            k parameter
        p: int
            Distance function, p=1: Manhattan, p=2: Euclidean
        """
        super(D_LOF).__init__()
        self.contamination: float = contamination
        self.n_neighbors: int = n_neighbors
        self.p: int = p
        self.clf: LOF = LOF(contamination=contamination, n_neighbors=n_neighbors, p=p, novelty=True)

    @property
    def info(self) -> dict:
        """Additional classifier information

        Returns
        -------
        dict
        """
        return {'Neighbors': self.n_neighbors, 'p': self.p, 'NoveltyThreshold': self.threshold}


class Detector_SKLearn(AbstractDetector, ABC):
    """Abstract base classifier for outlier detection with sk learn methods"""

    def __init__(self):
        super(Detector_SKLearn).__init__()
        self.decision_scores_ = None
        self.x_train_UnNormalized = None

    def train(self, x_train: ndarray):
        """Train classifier with training data

        Parameters
        ----------
        x_train: ndarray
            NxD matrix of training data, N: number of data points, D: number of dimensions
        """
        self.x_train_UnNormalized = x_train
        super().train(x_train)

    def get_decision_scores(self) -> ndarray:
        """Get novelty scores of training data

        Returns
        -------
        ndarray
            Novelty scores of training data; Nx1 matrix, N: number of data points
        """
        if self.decision_scores_ is None:
            self.decision_scores_ = self.score(self.x_train_UnNormalized)
        return self.decision_scores_

    def score(self, x_test: ndarray) -> ndarray:
        """Get novelty scores for training data

        Parameters
        ----------
        x_test: ndarray
            Test data to be scored: NxD matrix, N: number of data points, D: number of dimensions
        Returns
        -------
        ndarray
            Novelty scores;  Nx1 matrix, N: number of data points
        """
        x_test_c = self.norm(x_test)
        return self.clf.score_samples(x_test_c)

    def fit(self, x_train: ndarray):
        """Train classifier with training data

        Parameters
        ----------
        x_train: ndarray
            NxD matrix of training data, N: number of data points, D: number of dimensions
        """
        self.train(x_train)

    def decision_function(self, x_test: ndarray) -> ndarray:
        """Get novelty scores for training data

        Parameters
        ----------
        x_test: ndarray
            Test data to be scored: NxD matrix, N: number of data points, D: number of dimensions
        Returns
        -------
        ndarray
            Novelty scores;  Nx1 matrix, N: number of data points
        """
        return self.score(x_test)


class D_ParzenWindow(Detector_SKLearn):
    """Outlier detection with parzen window estimation (kernel density estimation)"""

    def __init__(self, contamination: float = 0.01, bandwith: float = 0.1, kernel: str = 'gaussian'):
        """

        Parameters
        ----------
        contamination: float
            PyOD specific: Expected fraction of outliers
        bandwith: float
            Bandwith parameter
        kernel: str
            Kernel parameter
        """
        super(D_ParzenWindow).__init__()
        self.bandwith: float = bandwith
        self.kernel: str = kernel
        self.contamination: float = contamination
        self.decision_scores_ = None
        self.clf: KernelDensity = KernelDensity(kernel=kernel, bandwidth=bandwith)

    def score(self, x_test: ndarray) -> ndarray:
        """Get novelty scores for training data

        Parameters
        ----------
        x_test: ndarray
            Test data to be scored: NxD matrix, N: number of data points, D: number of dimensions
        Returns
        -------
        ndarray
            Novelty scores;  Nx1 matrix, N: number of data points
        """
        scores = super().score(x_test)
        return -np.exp(scores)

    @property
    def info(self) -> dict:
        """Additional classifier information

        Returns
        -------
        dict
        """
        return {'Bandwith': self.bandwith, 'Kernel': self.kernel, 'NoveltyThreshold': self.threshold}


class D_GMM(Detector_SKLearn):
    """Gaussian Mixture Model"""

    def __init__(self, contamination: float = 0.01, n_components: int = 5):
        """

        Parameters
        ----------
        contamination: float
            PyOD specific: Expected fraction of outliers
        n_components: int
            Number of components parameter
        """
        super(D_GMM).__init__()
        self.n_components: int = n_components
        self.contamination: float = contamination
        self.decision_scores_ = None
        self.clf: GaussianMixture = GaussianMixture(n_components=n_components)

    def score(self, x_test: ndarray) -> ndarray:
        """Get novelty scores for training data

        Parameters
        ----------
        x_test: ndarray
            Test data to be scored: NxD matrix, N: number of data points, D: number of dimensions
        Returns
        -------
        ndarray
            Novelty scores;  Nx1 matrix, N: number of data points
        """
        scores = super().score(x_test)
        return -np.exp(scores)

    @property
    def info(self) -> dict:
        """Additional classifier information

        Returns
        -------
        dict
        """
        return {'N-Components': self.n_components, 'NoveltyThreshold': self.threshold}


class NoveltyDetectionGPR(GaussianProcessRegressor):
    """Gaussian process regression for outlier detection"""

    @ignore_warnings(category=ConvergenceWarning)
    def fit(self, x_train: ndarray):
        """Train classifier with training data

        Parameters
        ----------
        x_train: ndarray
            NxD matrix of training data, N: number of data points, D: number of dimensions
        """
        y_train = np.ones((x_train.shape[0], 1))
        super().fit(x_train, y_train)


class D_GP(AbstractDetector):
    """Outlier detection with gaussian process regression"""

    def __init__(self, contamination: float = 0.01,
                 kernel=1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))):
        """

        Parameters
        ----------
        contamination: float
            PyOD specific: Expected fraction of outliers
        kernel
            Kernel parameter
        """
        super(D_GP).__init__()
        self.contamination: float = contamination
        self.decision_scores_ = None
        self.kernel = kernel
        self.clf: NoveltyDetectionGPR = NoveltyDetectionGPR(kernel=kernel, random_state=0)
        self.x_train_UnNormalized = None

    def train(self, x_train: ndarray):
        """Train classifier with training data

        Parameters
        ----------
        x_train: ndarray
            NxD matrix of training data, N: number of data points, D: number of dimensions
        """
        self.x_train_UnNormalized = x_train
        super().train(x_train)

    def get_decision_scores(self) -> ndarray:
        """Get novelty scores of training data

        Returns
        -------
        ndarray
            Novelty scores of training data; Nx1 matrix, N: number of data points
        """
        if self.decision_scores_ is None:
            self.decision_scores_ = self.score(self.x_train_UnNormalized)
        return self.decision_scores_

    def score(self, x_test: ndarray) -> ndarray:
        """Get novelty scores for training data

        Parameters
        ----------
        x_test: ndarray
            Test data to be scored: NxD matrix, N: number of data points, D: number of dimensions
        Returns
        -------
        ndarray
            Novelty scores;  Nx1 matrix, N: number of data points
        """
        x_test_c = self.norm(x_test)
        y_mean, y_std = self.clf.predict(x_test_c, return_std=True)
        return y_std

    def fit(self, x_train: ndarray):
        """Train classifier with training data

        Parameters
        ----------
        x_train: ndarray
            NxD matrix of training data, N: number of data points, D: number of dimensions
        """
        self.train(x_train)

    def decision_function(self, x_test: ndarray):
        """Get novelty scores for training data

        Parameters
        ----------
        x_test: ndarray
            Test data to be scored: NxD matrix, N: number of data points, D: number of dimensions
        Returns
        -------
        ndarray
            Novelty scores;  Nx1 matrix, N: number of data points
        """
        return self.score(x_test)

    @property
    def info(self) -> dict:
        """Additional classifier information

        Returns
        -------
        dict
        """
        return {'LengthScale': self.clf.kernel.k2.length_scale, 'NoveltyThreshold': self.threshold}


class D_None(AbstractDetector):
    """Classifier always returning normal class"""

    def __init__(self, **kwargs):
        super(D_None).__init__()
        self.decision_scores_ = None
        self.threshold = 0

    def train(self, x_train: ndarray):
        """Train classifier with training data

        Parameters
        ----------
        x_train: ndarray
            NxD matrix of training data, N: number of data points, D: number of dimensions
        """
        self.decision_scores_ = -1 * np.ones(len(x_train))

    def norm(self, x_t: ndarray, init: bool = False):
        """Normalize data, min/max normalization

        Parameters
        ----------
        x_t: ndarray
            NxD matrix of data to be normalized, N: number of data points, D: number of dimensions
        init: bool
            If true, min/max values for normalization will be initialized

        Returns
        -------
        ndarray
            Normalized data, NxD matrix, N: number of data points, D: number of dimensions
        """
        return None

    def predict(self, x_test: ndarray) -> ndarray:
        """Classify test data

        Parameters
        ----------
        x_test: ndarray
            Test data to be classified: NxD matrix, N: number of data points, D: number of dimensions
        Returns
        -------
        ndarray
            Classification; 0: Normal data, 1: Outlier;  Nx1 matrix, N: number of data points
        """
        classification = np.zeros(len(x_test))
        return classification

    def score(self, x_test: ndarray) -> ndarray:
        """Get novelty scores for training data

        Parameters
        ----------
        x_test: ndarray
            Test data to be scored: NxD matrix, N: number of data points, D: number of dimensions
        Returns
        -------
        ndarray
            Novelty scores;  Nx1 matrix, N: number of data points
        """
        scores = -1 * np.ones(len(x_test))
        return scores

    def get_decision_scores(self) -> ndarray:
        """Get novelty scores of training data

        Returns
        -------
        ndarray
            Novelty scores of training data; Nx1 matrix, N: number of data points
        """
        return self.decision_scores_

    def get_clf(self, *args, **kwargs) -> AbstractDetector:
        """Return classifier

        Returns
        -------
        AbstractDetector
            returns classifier
        """
        return self

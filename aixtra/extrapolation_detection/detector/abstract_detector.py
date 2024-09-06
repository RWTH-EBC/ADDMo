from abc import ABC

import numpy as np
from numpy import ndarray


class AbstractDetector(ABC):
    """Abstract base novelty detection classifier"""

    def __init__(self):
        self.min = None
        self.max = None
        self.clf = None
        self.threshold = None

    def train(self, x_train: ndarray):
        """Train classifier with training system_data

        Parameters
        ----------
        x_train: ndarray
            NxD matrix of training system_data, N: number of system_data points, D: number of dimensions
        """
        # Copy system_data
        # x_train_c = x_train.copy()

        # Normalize training system_data
        x_train_c = self.norm(x_train, init=True)

        # Train classifier
        self.clf.fit(x_train_c)

    def norm(self, x_t: ndarray, init: bool = False) -> ndarray:
        """Normalize system_data, min/max normalization



        Parameters
        ----------
        x_t: ndarray
            NxD matrix of system_data to be normalized, N: number of system_data points, D: number of dimensions
        init: bool
            If true, min/max values for normalization will be initialized

        Returns
        -------
        ndarray
            Normalized system_data, NxD matrix, N: number of system_data points, D: number of dimensions
        """
        # Copy system_data
        x_t_c = x_t.copy()

        x_t_n = np.zeros(x_t_c.shape)
        nc = len(x_t_c[0])

        # Initialize min/max values
        if init:
            self.min: ndarray = np.zeros(nc)
            self.max: ndarray = np.zeros(nc)

        # Normalization per dimension
        for c in range(0, nc):

            # Initialize min/max values
            if init:
                self.min[c] = np.amin(x_t_c[:, c])
                self.max[c] = np.amax(x_t_c[:, c])

                # Fallback, if min/max are equal
                if self.min[c] == self.max[c]:
                    self.min[c] = 0
                    self.max[c] = 1
                    print('Warning: Normalization failed.')

            # Min/Max normalization
            x_t_n[:, c] = (x_t_c[:, c] - self.min[c]) / (self.max[c] - self.min[c])
        return x_t_n

    def predict(self, x_test: ndarray) -> ndarray:
        """Classify test system_data

        Parameters
        ----------
        x_test: ndarray
            Test system_data to be classified: NxD matrix, N: number of system_data points, D: number of dimensions
        Returns
        -------
        ndarray
            Classification; 0: Normal system_data, 1: Outlier;  Nx1 matrix, N: number of system_data points
        """
        # Get novelty scores
        scores = self.score(x_test)

        # Classify system_data
        classification = np.zeros(len(x_test))
        classification[scores > self.threshold] = 1
        return classification

    def score(self, x_test: ndarray) -> ndarray:
        """Get novelty scores for training system_data

        Parameters
        ----------
        x_test: ndarray
            Test system_data to be scored: NxD matrix, N: number of system_data points, D: number of dimensions
        Returns
        -------
        ndarray
            Novelty scores;  Nx1 matrix, N: number of system_data points
        """
        # Normalize system_data
        x_test_c = self.norm(x_test)

        # Return decision scores
        return self.clf.decision_function(x_test_c)

    def get_decision_scores(self) -> ndarray:
        """Get novelty scores of training system_data

        Returns
        -------
        ndarray
            Novelty scores of training system_data; Nx1 matrix, N: number of system_data points
        """
        return self.clf.decision_scores_

    @property
    def info(self) -> dict:
        """Additional classifier information

        Returns
        -------
        dict
        """
        return {}

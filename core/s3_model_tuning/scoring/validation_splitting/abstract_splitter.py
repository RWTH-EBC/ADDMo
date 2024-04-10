import numpy as np

import pandas as pd

from sklearn.model_selection import BaseCrossValidator


class AbstractSplitter(BaseCrossValidator):
    """
    Generate a splitter that is compatible with scikit-learn cross-validation tools.

    Split: On a CV splitter (not an estimator), this method accepts parameters (X, y, groups),
    where all may be optional, and returns an iterator over (train_idx, test_idx) pairs. Each of {
    train,test}_idx is a 1d integer array, with values from 0 from X.shape[0] - 1 of any length,
    such that no values appear in both some train_idx and its corresponding test_idx.

    cross-validation generator: A non-estimator family of classes used to split a dataset into a
    sequence of train and test portions (see Cross-validation: evaluating estimator performance),
    by providing split and get_n_splits methods. Note that unlike estimators, these do not have fit
    methods and do not provide set_params or get_params. Parameter validation may be performed in
    __init__."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def split(self, X: pd.DataFrame = None, y: pd.Series = None, groups=None):
        """Generate indices to split data into training and test sets. This dummy implementation
        is copied from scikit-learn. It ensures that the train set always contains the remaining
        indices compared to the test set. If you don't want this behavior, you can override this
        method in your custom splitter. Otherwise, I recommend keeping this method as it is and
        making changes to the _iter_test_indices method.
        """
        indices = np.arange(len(X))
        for test_index in self._iter_test_masks(X, y, groups):
            train_index = indices[np.logical_not(test_index)]
            test_index = indices[test_index]
            yield train_index, test_index

    def get_n_splits(self, X: pd.DataFrame = None, y: pd.Series = None, groups=None):
        """Return the number of splitting iterations in the cross-validator."""
        raise NotImplementedError

    def _iter_test_masks(
        self, X: pd.DataFrame = None, y: pd.Series = None, groups=None
    ):
        """Generates boolean masks corresponding to test sets.

        By default, delegates to _iter_test_indices(X, y, groups)
        """
        for test_index in self._iter_test_indices(X, y, groups):
            test_mask = np.zeros(len(X), dtype=bool)
            test_mask[test_index] = True
            yield test_mask

    def _iter_test_indices(
        self, X: pd.DataFrame = None, y=None, groups=None
    ) -> np.ndarray:
        """Generates integer indices corresponding to test sets."""
        raise NotImplementedError

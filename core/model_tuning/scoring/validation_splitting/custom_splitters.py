from sklearn.model_selection import KFold

from core.model_tuning.scoring.validation_splitting.abstract_splitter import AbstractSplitter

'''Creating custom splitter that work with scikit-learn.
An iterable yielding (train, test) splits as arrays of indices.'''

class NoSplitting(AbstractSplitter):
    def get_n_splits(self, X=None, y=None, groups=None):
        return 1

class myKFold(KFold):
    def nur_zum_testen_ob_das_klappt(self):
        pass

class TrialCustomSplitter(AbstractSplitter):
    """Custom splitter for scikit-learn cross-validation.

    This splitter creates two folds:
    - The first fold includes the first and last 10 rows of the dataset.
    - The second fold includes the rows in between.
    """

    def get_n_splits(self, X=None, y=None, groups=None):
        """Return the number of splitting iterations in the cross-validator."""
        return 2

    def _iter_test_indices(self, X, y=None, groups=None):
        """Generate integer indices for test set for each fold."""
        n_samples = len(X)
        if n_samples < 20:
            raise ValueError("The dataset must have at least 20 samples for this custom splitter.")

        first_fold_indices = list(range(10)) + list(range(n_samples - 10, n_samples))
        second_fold_indices = list(range(10, n_samples - 10))

        yield first_fold_indices
        yield second_fold_indices


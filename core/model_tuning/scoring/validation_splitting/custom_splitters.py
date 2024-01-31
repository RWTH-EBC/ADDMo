from sklearn.model_selection import KFold

from core.model_tuning.scoring.validation_splitting.abstract_splitter import AbstractSplitter

'''Creating custom splitter that work with scikit-learn. Please see the documentation of the 
AbstractSplitter class for more information.'''

class TrialCustomSplitter(AbstractSplitter):
    """Custom splitter for scikit-learn cross-validation.

    This splitter creates two folds:
    - The first fold includes the first and last 10 rows of the dataset.
    - The second fold includes the first 100 rows.

    Both folds are used once as test set and once as train set (due to cross-validation).

    This splitter is only for demonstration purposes and should not be used in production.
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
        yield first_fold_indices

        second_fold_indices = list(range(0, 100))
        yield second_fold_indices
        # if you only yield one fold the cross-validation will only produce one score on the
        # yielded test indices. For each yielded fold, e.g. through a for loop, the cross-validation
        # will produce one score on the yielded test indices.




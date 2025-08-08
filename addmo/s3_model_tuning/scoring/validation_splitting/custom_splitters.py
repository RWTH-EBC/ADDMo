from addmo.s3_model_tuning.scoring.validation_splitting.abstract_splitter import (
    AbstractSplitter,
)

"""Creating custom splitter that work with scikit-learn. Please see the documentation of the 
AbstractSplitter class for more information."""


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
            raise ValueError(
                "The dataset must have at least 20 samples for this custom splitter."
            )

        first_fold_indices = list(range(10)) + list(range(n_samples - 10, n_samples))
        yield first_fold_indices

        second_fold_indices = list(range(0, 100))
        yield second_fold_indices
        # if you only yield one fold the cross-validation will only produce one score on the
        # yielded test indices. For each yielded fold, e.g. through a for loop, the cross-validation
        # will produce one score on the yielded test indices.


class UnivariateSplitter(AbstractSplitter):
    """
    This class inherits from `AbstractSplitter` and is designed to split datasets along a single
    feature dimension based on predefined ratios. Creates one split for each feature in the
    dataset, where each test set is composed of system_data points from the top, bottom, and middle
    sections of the sorted feature values.
    """

    def __init__(
        self, top_split_ratio=0.1, bottom_split_ratio=0.1, middle_split_ratio=0.1
    ):
        # Ensure the sum of provided ratios does not exceed 1
        total_ratio = sum(
            filter(None, [top_split_ratio, bottom_split_ratio, middle_split_ratio])
        )
        if total_ratio > 1:
            raise ValueError("The sum of all split ratios must not exceed 1.")

        self.top_split_ratio = top_split_ratio
        self.bottom_split_ratio = bottom_split_ratio
        self.middle_split_ratio = middle_split_ratio

    def get_n_splits(self, X, y=None, groups=None):
        if X is None:
            raise ValueError("The 'X' parameter should not be None.")
        return X.shape[1]

    def _iter_test_indices(self, X, y=None, groups=None):
        for feature_index in range(X.shape[1]):
            # Sort once for each feature
            sorted_indices = X.iloc[:, feature_index].sort_values(ascending=True).index

            test_indices = []
            if self.top_split_ratio:
                num_top_tests = int(X.shape[0] * self.top_split_ratio)
                test_indices.extend(sorted_indices[-num_top_tests:])

            if self.bottom_split_ratio:
                num_bottom_tests = int(X.shape[0] * self.bottom_split_ratio)
                test_indices.extend(sorted_indices[:num_bottom_tests])

            if self.middle_split_ratio:
                total_middle_tests = int(X.shape[0] * self.middle_split_ratio)
                start_index = (X.shape[0] - total_middle_tests) // 2
                end_index = start_index + total_middle_tests
                test_indices.extend(sorted_indices[start_index:end_index])

            # Ensure unique indices in case of overlap
            unique_test_indices = list(set(test_indices))

            test_indices_positions = [
                X.index.get_loc(idx) for idx in unique_test_indices
            ]
            yield test_indices_positions

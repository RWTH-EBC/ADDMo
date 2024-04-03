import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from itertools import combinations
import pandas as pd

from extrapolation_detection.detector.abstract_detector import AbstractDetector
from extrapolation_detection.detector.detector_factory import DetectorFactory


def verify_bounds(x: pd.DataFrame, bounds: dict):
    """Verify that the data set does not contain values outside of the boundaries."""
    # check if all variables in the data set are present in the boundaries
    for variable in x.columns:
        if variable not in bounds:
            raise ValueError(f"Variable {variable} is not present in the boundaries.")

    # check if data set contains values outside of the boundaries
    for variable in x.columns:
        min_val, max_val = bounds[variable]
        if x[variable].min() < min_val:
            print(
                f"Variable {variable} contains values below the minimum boundary ({min_val}):"
            )
            print(x[x[variable] < min_val])
        if x[variable].max() > max_val:
            print(
                f"Variable {variable} contains values above the maximum boundary ({max_val}):"
            )
            print(x[x[variable] > max_val])


def clip_out_of_bound_values(x: pd.DataFrame, bounds: dict):
    """Clip values outside of the boundaries to the boundaries."""
    for variable in x.columns:
        min_val, max_val = bounds[variable]
        x[variable] = x[variable].clip(min_val, max_val)
    return x


class GridOccupancy:
    def __init__(self, grid_divisions):
        self.grid_divisions = grid_divisions
        self.boundaries = None
        self.occupancy_grid = None

    def train(self, dataset, boundaries):
        """
        Determines occupied grid cells based on the dataset.

        Parameters:
        - dataset: numpy array of shape (n_samples, n_features)
        - boundaries: dict, {variable_name: (min, max)}
        """
        verify_bounds(dataset, boundaries)
        dataset = clip_out_of_bound_values(dataset, boundaries)

        self.boundaries = boundaries
        num_features = dataset.shape[1]
        grid_shape = [self.grid_divisions] * num_features
        self.occupancy_grid = np.zeros(grid_shape, dtype=int)

        # Normalize dataset within boundaries
        for var in dataset.columns:
            min_val, max_val = boundaries[var]
            # Normalize dataset directly to grid indices range
            dataset[var] = (
                (dataset[var] - min_val)
                / (max_val - min_val)
                * (self.grid_divisions - 1) # to account index starting from 0
            )
            dataset[var] = np.floor(dataset[var]).astype(int)

        # Mark grid cells as occupied
        for point in dataset.itertuples(index=False):
            self.occupancy_grid[point] = 1

    def calculate_coverage(self):
        """
        Calculates the coverage percentage of the grid.

        Returns:
        - coverage_percentage: float
        """
        total_cells = self.occupancy_grid.size
        occupied_cells = np.sum(self.occupancy_grid)
        coverage = (occupied_cells / total_cells) * 100
        coverage = pd.Series(
            (coverage, 100-coverage), index=["Inside", "Outside"]
        )
        return coverage


    def predict(self, new_point):
        """
        Predicts if a new point is an inlier (0) or an outlier (1).

        Parameters:
        - new_point: list or numpy array

        Returns:
        - prediction: int
        """
        for i, boundary in enumerate(self.boundaries.values()):
            if new_point[i] < boundary[0] or new_point[i] > boundary[1]:
                return 1  # Outlier

        normalized_new_point = np.empty(len(new_point))
        for i, (min_val, max_val) in enumerate(self.boundaries.values()):
            normalized_new_point[i] = (
                (new_point[i] - min_val) / (max_val - min_val) * self.grid_divisions
            )
            normalized_new_point[i] = np.floor(normalized_new_point[i]).astype(int)
            normalized_new_point[i] = np.clip(
                normalized_new_point[i], 0, self.grid_divisions - 1
            )

        return 0 if self.occupancy_grid[tuple(normalized_new_point)] else 1


class ExplorationQuantifier:
    def __init__(self):
        self.x_grid: pd.DataFrame = None
        self.labels: np.ndarray = None
        self.explo_clf = None

    def train_exploration_classifier(
        self, x_fit: pd.DataFrame, classifier_name="ConvexHull"
    ):
        clf: AbstractDetector = DetectorFactory.detector_factory(classifier_name)
        clf.train(x_fit.values)

        # Set the threshold to the maximum score
        if not classifier_name == "ConvexHull":
            scores = clf.score(x_fit.values)
            clf.threshold = np.max(scores)  # all points are inlier

        self.explo_clf = clf

    def calc_labels(self, x_grid):
        """Calculate the labels for the points using the exploration classifier.

        label = 0 means the point is inside, label = 1 means the point is outside."""
        self.x_grid = x_grid
        self.labels_grid = self.explo_clf.predict(self.x_grid.values)

        self.labels_grid = self.labels_grid.astype(int)
        self.labels_grid = pd.Series(self.labels_grid)

        return self.labels_grid

    def calculate_coverage(self):
        # Calculate the volume percentages
        coverage = self.labels_grid.value_counts(normalize=True) * 100

        # Handle situations where one labels misses completely (e.g. all data inside)
        coverage.loc["Inside"] = coverage.get(0, 0) # get the value if it exists, otherwise 0
        coverage.loc["Outside"] = coverage.get(1, 0) # get the value if it exists, otherwise 0

        return coverage

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import pandas as pd

from pyod.models.knn import KNN

from extrapolation_detection.detector.abstract_detector import AbstractDetector
from extrapolation_detection.detector.detector_factory import DetectorFactory




class ExplorationQuantifier:
    def __init__(self, x, points, bounds):
        self.x = x
        self.points: pd.DataFrame = points
        self.bounds: dict(tuple) = bounds
        self.labels: np.ndarray = None

    def train_exploration_classifier(self, classifier_name="ConvexHull"):
        clf: AbstractDetector = DetectorFactory.detector_factory(classifier_name)
        clf.train(self.x.values)

        # Set the threshold to the maximum score
        if not classifier_name == "ConvexHull":
            scores = clf.score(self.x.values)
            clf.threshold = np.max(scores) # all points are inlier

        self.explo_clf = clf

    def calc_labels(self):
        '''Calculate the labels for the points using the exploration classifier.

        label = 0 means the point is inside, label = 1 means the point is outside.'''

        self.points_labeled = self.explo_clf.predict(self.points.values)

        self.points_labeled = self.points_labeled.astype(int)
        self.points_labeled = pd.Series(self.points_labeled)

        return self.points_labeled

    def calculate_exploration_percentages(self):
        # Count the number of points that fall into each region
        region_counts = np.bincount(self.points_labeled.values)

        # Calculate the volume percentages
        exploration_percentages = region_counts / len(self.points) * 100
        exploration_percentages = pd.Series(exploration_percentages, index=["Inside", "Outside"])

        return exploration_percentages

    def plot_scatter_extrapolation_share_2D(self, explo_detector_name):
        # Get all combinations of variables
        variable_combinations = combinations(self.points.columns, 2)

        # Create a DataFrame that contains the generated points and their labels
        points_df = pd.DataFrame(self.points, columns=self.points.columns)
        points_df['label'] = self.points_labeled

        for var1, var2 in variable_combinations:
            # Group by var1 and var2 and calculate the mean of the label
            average_labels = points_df.groupby([var1, var2])['label'].mean()

            # Reset the index to convert var1 and var2 back into columns
            average_labels_df = average_labels.reset_index()

            # Create a scatter plot for each combination of variables
            plt.figure(figsize=(5, 5))
            plt.scatter(average_labels_df[var1], average_labels_df[var2], c=average_labels_df[
                'label'], cmap='viridis', vmin=0, vmax=1)

            plt.xlabel(var1)
            plt.ylabel(var2)
            plt.title(f'{explo_detector_name}\n{var1} and {var2}')
            plt.colorbar(label='Extrapolation state\n averaged over the remaining dimensions (%)')
            yield plt

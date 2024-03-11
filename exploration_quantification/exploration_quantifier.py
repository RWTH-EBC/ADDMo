import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import pandas as pd

from extrapolation_detection.detector.abstract_detector import AbstractDetector
from extrapolation_detection.detector.detector_factory import DetectorFactory



class ExplorationQuantifier:
    def __init__(self, points, bounds):
        self.points: pd.DataFrame = points
        self.bounds: dict(tuple) = bounds
        self.labels: np.ndarray = None

    def train_exploration_classifier(self, classifier_name="D_ConvexHull"):
        clf = DetectorFactory.detector_factory(classifier_name)
        clf.train(self.x)
        self.explo_clf = clf
    def calc_labels(self, points):
        '''Calculate the labels for the points using the exploration classifier.

        label = 0 means the point is in '''
        points_labeled = self.explo_clf.predict(points)

        return points_labeled

    def calculate_exploration_percentages(self, classifier):
        # Apply the classifier to the random points
        labels = classifier.predict(self.points.values())

        # Shift labels to ensure they are non-negative
        labels = labels - labels.min()

        # Count the number of points that fall into each region
        region_counts = np.bincount(labels, minlength=2)

        # Calculate the volume percentages
        region_percentages = region_counts / len(self.points) * 100

        self.labels = labels
        return region_percentages

    def plot_scatter_extrapolation_share_2D(self):
        # Get all combinations of variables
        variable_combinations = combinations(self.points.columns, 2)

        # Create a DataFrame that contains the generated points and their labels
        points_df = pd.DataFrame(self.points, columns=self.points.columns)
        points_df['label'] = self.labels

        for var1, var2 in variable_combinations:
            plt.figure(figsize=(5, 5))

            # Group by var1 and var2 and calculate the mean of the label
            average_labels = points_df.groupby([var1, var2])['label'].mean()

            # Reset the index to convert var1 and var2 back into columns
            average_labels_df = average_labels.reset_index()

            # Create a scatter plot for each combination of variables
            plt.scatter(average_labels_df[var1], average_labels_df[var2], c=average_labels_df[
                'label'], cmap='viridis', vmin=0, vmax=1)

            plt.xlabel(var1)
            plt.ylabel(var2)
            plt.title(f'{var1} and {var2}')
            plt.colorbar(label='Extrapolation state\n averaged over the remaining dimensions (%)')

            plt.show()

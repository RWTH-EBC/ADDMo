import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import pandas as pd
class ExplorationQuantifier:
    def __init__(self, xy, xy_boundaries):
        self.xy = xy
        self.xy_boundaries = xy_boundaries
        self.points = None

    def number_of_points_per_variable_2_total(self, num_points_per_variable):
        # Infer the number of points from the number of variables
        num_points = np.prod(num_points_per_variable)
        return num_points

    def generate_random_points(self, num_points_per_variable=100):
        num_points = np.prod(num_points_per_variable)
        random_points = np.array(
            [
                np.random.uniform(
                    low=self.xy_boundaries[var][0], high=self.xy_boundaries[var][1], size=num_points
                )
                for var in self.xy.columns
            ]
        ).T
        self.points = random_points
        return random_points

    def generate_point_grid(self, num_points_per_variable):
        # Generate a grid of points within the specified boundaries for each variable
        grids = np.meshgrid(*[
            np.linspace(start=self.xy_boundaries[var][0], stop=self.xy_boundaries[var][1],
                        num=num_points_per_variable)
            for var in self.xy.columns
        ])

        # Reshape and stack the grids to get a single array of points
        grid_points = np.vstack([grid.ravel() for grid in grids]).T

        self.points = grid_points
        return grid_points


    def calculate_exploration_percentages(self, classifier):
        # Apply the classifier to the random points
        labels = classifier.predict(self.points)

        # Shift labels to ensure they are non-negative
        labels = labels - labels.min()

        # Count the number of points that fall into each region
        region_counts = np.bincount(labels, minlength=2)

        # Calculate the volume percentages
        region_percentages = region_counts / len(self.points) * 100

        self.labels = labels
        return region_percentages

    def plot_scatter(self):
        # Get all combinations of variables
        variable_combinations = combinations(self.xy.columns, 2)

        # Create a DataFrame that contains the generated points and their labels
        points_df = pd.DataFrame(self.points, columns=self.xy.columns)
        points_df['label'] = self.labels

        for var1, var2 in variable_combinations:
            plt.figure(figsize=(10, 10))

            # Create a scatter plot for each combination of variables
            plt.scatter(points_df[var1], points_df[var2], c=points_df['label'], cmap='viridis')

            plt.xlabel(var1)
            plt.ylabel(var2)
            plt.title(f'Classes for {var1} and {var2}')
            plt.colorbar(label='Class')

            plt.show()
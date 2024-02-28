import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import pandas as pd


class ArtificialPointGenerator:

    @staticmethod
    def infer_meshgrid_bounds(df: pd.DataFrame):
        '''Infer the boundaries of the meshgrid from the dataframe.
        The boundaries are used to generate the meshgrid for the 2D plot.
        Df should contain the variables to be gridded over.
        '''

        # Get bounds of nD plot
        bounds = {}
        for column in df.columns:
            min_val = df[column].min()
            max_val = df[column].max()
            bounds[column] = (min_val, max_val)
        return bounds

    @staticmethod
    def generate_random_points(df, bounds, num_points_per_variable=100):
        num_points = np.prod(num_points_per_variable)
        random_points = np.array(
            [
                np.random.uniform(
                    low=bounds[var][0], high=bounds[var][1],
                    size=num_points
                )
                for var in df.columns
            ]
        ).T
        random_points_df = pd.DataFrame(random_points, columns=df.columns)
        return random_points_df

    @staticmethod
    def generate_point_grid(df, bounds, num_points_per_variable=100):
        # Generate a grid of points within the specified boundaries for each variable
        grids = np.meshgrid(*[
            np.linspace(start=bounds[var][0], stop=bounds[var][1],
                        num=num_points_per_variable, dtype=np.float32)
            for var in df.columns
        ])

        # Reshape and stack the grids to get a single array of points
        grid_points = np.vstack([grid.ravel() for grid in grids]).T

        grid_points_df = pd.DataFrame(grid_points, columns=df.columns)
        return grid_points_df

class ExplorationQuantifier(ArtificialPointGenerator):
    def __init__(self, xy, xy_boundaries):
        self.xy = xy
        self.xy_boundaries: dict(tuple) = xy_boundaries
        self.points = None

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

    def plot_scatter_extrapolation_share_2D(self):
        # Get all combinations of variables
        variable_combinations = combinations(self.xy.columns, 2)

        # Create a DataFrame that contains the generated points and their labels
        points_df = pd.DataFrame(self.points, columns=self.xy.columns)
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

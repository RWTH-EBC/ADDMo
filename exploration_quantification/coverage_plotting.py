import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

from pandas.plotting import parallel_coordinates
from pandas.plotting import scatter_matrix
import plotly.express as px

def plot_scatter_average_coverage_per_2D(
    x_grid: pd.DataFrame, y_grid: pd.DataFrame, title_header: str
):
    """Plot a scatter plot for each combination of variables in the x_grid DataFrame."""
    # Get all combinations of variables
    variable_combinations = combinations(x_grid.columns, 2)

    # Create a DataFrame that contains the generated points and their labels
    xy_grid = x_grid.copy()
    xy_grid["label"] = y_grid

    for var1, var2 in variable_combinations:
        # Group by var1 and var2 and calculate the mean of the label
        average_labels = xy_grid.groupby([var1, var2])["label"].mean()

        # Reset the index to convert var1 and var2 back into columns
        average_labels_df = average_labels.reset_index()

        # Create a scatter plot for each combination of variables
        plt.figure(figsize=(5, 5))
        plt.scatter(
            average_labels_df[var1],
            average_labels_df[var2],
            c=average_labels_df["label"],
            cmap="viridis",
            vmin=0,
            vmax=1,
        )

        plt.xlabel(var1)
        plt.ylabel(var2)
        plt.title(f"{title_header}\n{var1} and {var2}")
        plt.colorbar(
            label="Extrapolation state\n averaged over the remaining dimensions (%)"
        )
        yield plt


def plot_grid_cells_average_coverage_per_2D(
    coverage_grid: np.ndarray,
    boundaries: dict,
    variable_names: list[str],
    grid_cells_per_axis: int,
    title_header: str,
):
    """
    Plots the grid cells, highlighting occupied ones.
    """
    # Get all combinations of variables
    variable_combinations = combinations(range(coverage_grid.ndim), 2)

    for var1, var2 in variable_combinations:
        # average the grid cells over the remaining dimensions
        axes = tuple(set(range(coverage_grid.ndim)) - {var1, var2})
        averaged_grid = np.mean(coverage_grid, axis=axes) * 100

        # plot this averaged grid occupancy for var1 and var2
        var1_name = variable_names[var1]
        var2_name = variable_names[var2]

        plt.figure()
        plt.imshow(averaged_grid, origin="lower")  # , vmin=0, vmax=1)
        plt.title(f"{title_header}\n{var1_name} and {var2_name}")

        # Set the ticks and labels to correspond to the original variable values not the normalized ones
        min_val1, max_val1 = boundaries[var1_name]
        min_val2, max_val2 = boundaries[var2_name]
        plt.xticks(
            np.linspace(0, grid_cells_per_axis - 1, 5),
            np.linspace(min_val1, max_val1, 5),
        )
        plt.yticks(
            np.linspace(0, grid_cells_per_axis - 1, 5),
            np.linspace(min_val2, max_val2, 5),
        )

        plt.xlabel(f"{var1_name}")
        plt.ylabel(f"{var2_name}")
        plt.colorbar(
            label="Extrapolation state\n averaged over the remaining dimensions (%)"
        )
        yield plt


def plot_dataset_distribution_kde(
    x: pd.DataFrame, title_header: str
):
    """
    Plots the distribution of the dataset using kernel density estimation (KDE).
    """
    # g = sns.pairplot(x, diag_kind="kde")
    g = sns.pairplot(x, diag_kind="hist")

    # colorful heatmap
    # g.map_lower(sns.kdeplot, fill=True, thresh=0, levels=100, cmap="mako")

    # layer lines on top of the scatter
    # g.map_lower(sns.kdeplot, levels=4, color=".2")



    plt.suptitle(title_header)
    plt.show()

def plot_dataset_parallel_coordinates(
    x: pd.DataFrame, title_header: str
):
    # Create a temporary DataFrame with a dummy 'class' column because parallel_coordinates expects it
    temp_x = x.copy()
    temp_x['class'] = 0  # Adding a dummy class label

    # Create the parallel coordinates plot
    plt.figure(figsize=(12, 8))  # Optional: Adjust figure size as needed
    parallel_coordinates(temp_x, class_column='class', colormap=plt.get_cmap("viridis"), alpha=0.5)

    # Removing the dummy 'class' label from the legend, if not desired
    plt.legend().remove()

    # Setting the title of the plot
    plt.title(title_header)

    # Improving layout
    plt.xticks(rotation=90)  # Rotate x-axis labels if they overlap
    plt.tight_layout()  # Adjust layout to make room for the rotated x-axis labels

    plt.show()


def plot_dataset_parallel_coordinates_plotly(x: pd.DataFrame, title_header: str):
    """
    Plots the dataset using parallel coordinates with Plotly.

    Parameters:
    - x: A pandas DataFrame containing the dataset to be plotted. All columns should be float.
    - title_header: A string representing the title of the plot.
    """
    fig = px.parallel_coordinates(x, color_continuous_scale=px.colors.diverging.Tealrose,
                                  color_continuous_midpoint=2)
    fig.update_layout(title=title_header)
    fig.show()

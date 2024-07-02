import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

from pandas.plotting import parallel_coordinates
import plotly.graph_objects as go


def plot_scatter_average_coverage_per_2D(
    x_grid: pd.DataFrame, y_grid: pd.DataFrame, title_header: str
):
    """Plot a scatter plot for each combination of variables in the x_grid DataFrame."""
    # Get all combinations of variables
    variable_combinations = combinations(x_grid.columns, 2)

    # Create a DataFrame that contains the generated points and their labels
    xy_grid = x_grid.copy()
    xy_grid["label"] = y_grid

    for var2, var1 in variable_combinations: # reversed order to match the other plots
        # Group by var1 and var2 and calculate the mean of the label
        average_labels = xy_grid.groupby([var1, var2])["label"].mean()*100

        # Reset the index to convert var1 and var2 back into columns
        average_labels_df = average_labels.reset_index()

        # Turn notation around (100 = 100% coverage)
        average_labels_df["label"] = 100 - average_labels_df["label"]

        # Create a scatter plot for each combination of variables
        plt.figure(figsize=(5, 5))
        plt.scatter(
            average_labels_df[var2],
            average_labels_df[var1],
            c=average_labels_df["label"],
            cmap="viridis",
            vmin=0,
            vmax=100,
        )

        plt.xlabel(var2)
        plt.ylabel(var1)
        plt.title(f"{title_header}\n{var1} and {var2}")
        plt.colorbar(
            label="Coverage\n averaged over the remaining dimensions (%)"
        )
        plt.tight_layout()
        yield plt


def plot_grid_cells_average_coverage_per_2D(
    coverage_grid: np.ndarray,
    boundaries: dict,
    variable_names: list[str],
    title_header: str,
):
    """
    Plots the grid cells, highlighting occupied ones.
    """
    # Get all combinations of variables
    variable_combinations = combinations(range(coverage_grid.ndim), 2)
    grid_cells_per_axis = coverage_grid.shape[0]

    for var2, var1 in variable_combinations: # reversed order to match the order of the axes
        # average the grid cells over the remaining dimensions
        axes = tuple(set(range(coverage_grid.ndim)) - {var1, var2})
        averaged_grid = np.mean(coverage_grid, axis=axes) * 100
        # transpose to match the order of axes of the other plots
        averaged_grid = np.transpose(averaged_grid)

        # Get the variable names
        var1_name = variable_names[var1]
        var2_name = variable_names[var2]

        # plot this averaged grid occupancy for var1 and var2
        plt.figure()
        plt.imshow(averaged_grid, origin="lower") #, vmin=0, vmax=100)
        plt.title(f"{title_header}\n{var1_name} and {var2_name}")

        # Set the ticks and labels to correspond to the original variable values not the normalized ones
        min_val1, max_val1 = boundaries[var1_name]
        min_val2, max_val2 = boundaries[var2_name]

        def calibrate_axis_rounding(max_val):
            if max_val > 100:
                return 0
            elif max_val > 10:
                return 1
            elif max_val > 1:
                return 2
            else:
                return 3

        rounding_axis1 = calibrate_axis_rounding(max_val1)
        rounding_axis2 = calibrate_axis_rounding(max_val2)

        plt.yticks(
            np.linspace(0, grid_cells_per_axis - 1, 5),
            np.round(np.linspace(min_val1, max_val1, 5), rounding_axis1)
        )
        plt.xticks(
            np.linspace(0, grid_cells_per_axis - 1, 5),
            np.round(np.linspace(min_val2, max_val2, 5), rounding_axis2),
        )

        plt.ylabel(f"{var1_name}")
        plt.xlabel(f"{var2_name}")
        plt.colorbar(
            label="Coverage\n averaged over the remaining dimensions (%)"
        )
        yield plt


def plot_dataset_distribution_kde(x: pd.DataFrame, bounds: dict, title_header: str):
    """
    Plots the distribution of the dataset using kernel density estimation (KDE).
    """
    # g = sns.pairplot(x, diag_kind="kde")
    g = sns.pairplot(x, diag_kind="hist")

    # colorful heatmap
    # g.map_upper(sns.kdeplot, fill=True, thresh=0, levels=100, cmap="mako")

    # layer lines on top of the scatter
    g.map_upper(sns.kdeplot, levels=4, color=".2")

    # Iterate over the axes matrix of the PairGrid
    for i, j in zip(*np.tril_indices_from(g.axes, -1)):
        # Set x-axis limits for lower triangle
        g.axes[i, j].set_xlim(bounds[x.columns[j]])
        # Set y-axis limits for lower triangle and diagonal
        g.axes[i, j].set_ylim(bounds[x.columns[i]])

    # Iterate through the diagonal to set x-axis limits since diag_kind="hist"
    for k in range(len(x.columns)):
        g.axes[k, k].set_xlim(bounds[x.columns[k]])

    plt.suptitle(title_header)
    return plt


def plot_dataset_parallel_coordinates(x: pd.DataFrame, title_header: str):
    # Create a temporary DataFrame with a dummy 'class' column because parallel_coordinates expects it
    temp_x = x.copy()
    temp_x["class"] = 0  # Adding a dummy class label

    # Create the parallel coordinates plot
    plt.figure(figsize=(12, 8))  # Optional: Adjust figure size as needed
    parallel_coordinates(
        temp_x, class_column="class", colormap=plt.get_cmap("viridis"), alpha=0.5
    )

    # Removing the dummy 'class' label from the legend, if not desired
    plt.legend().remove()

    # Setting the title of the plot
    plt.title(title_header)

    # Improving layout
    plt.xticks(rotation=90)  # Rotate x-axis labels if they overlap
    plt.tight_layout()  # Adjust layout to make room for the rotated x-axis labels

    return plt



def plot_dataset_parallel_coordinates_plotly(
    x: pd.DataFrame, bounds: dict, title_header: str
):
    """
    Plots the dataset using parallel coordinates with Plotly, allowing for initial constrained ranges.
    """
    # Prepare the dimensions for the parallel coordinates plot
    dimensions = []
    for col in x.columns:
        dim = {
            "range": [bounds[col][0], bounds[col][1]],
            "label": col,
            "values": x[col],
        }
        dimensions.append(dim)

    # Create the figure with parallel coordinates
    fig = go.Figure(
        data=go.Parcoords(
            dimensions=dimensions,
            line=dict(
                color="blue",  # Line color; customize as needed
            ),
        )
    )

    fig.update_layout(title=title_header, plot_bgcolor="white", paper_bgcolor="white")

    return fig

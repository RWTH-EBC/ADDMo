import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from addmo.util import plotting as d
from addmo.util.definitions import  return_results_dir_model_tuning, return_best_model
from addmo.s3_model_tuning.models.model_factory import ModelFactory
from addmo.util.load_save import load_data
import matplotlib.pyplot as plt
from itertools import product
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import ListedColormap
from addmo.util.plotting import *

from matplotlib.patches import Patch


def plot_carpets(model_config, bounds= None, combinations=None, defaults_dict=None, path_to_regressor = None):
    """
    Create 3D surface model_plots for prediction function.
    """

    target = model_config["name_of_target"]
    # Load regressor
    if path_to_regressor is None:
        path_to_regressor = return_best_model(return_results_dir_model_tuning(model_config['name_of_raw_data'],
                                                                              model_config['name_of_data_tuning_experiment'],
                                                                              model_config[ 'name_of_model_tuning_experiment']))
    regressor = ModelFactory.load_model(path_to_regressor)
    # Do not use the input data if user provides bounds and default dictionary
    if bounds is not None and defaults_dict is not None:
        variables = regressor.metadata["features_ordered"]

    # Load the input data and fetch column names as well as bounds from it
    else:
        data_path = model_config['abs_path_to_data']
        data = load_data(data_path)
        # Fetch time column from dataset
        time_column = next((col for col in data.columns if pd.api.types.is_datetime64_any_dtype(data[col])), None)
        # Set time column as index
        if time_column:
            data.set_index(time_column, inplace=True)
        x_grid = data.drop(target, axis=1)
        variables = list(x_grid.columns)

    # Define bounds
    if bounds is None:
        bounds = {}
        for var in variables:
            if var in x_grid.columns:
                bounds[var] = [x_grid[var].min(), x_grid[var].max()]

    # Define default values
    if defaults_dict is None:
        # Use mean value as default
        defaults_dict = {var: x_grid[var].mean() for var in variables}

    # Create combinations
    if combinations is None:
        combinations = [
            (v1, v2) for i, v1 in enumerate(variables) for v2 in variables[i + 1:]
        ]

    prediction_func= prediction_func_4_regressor(regressor)

    # Create a grid for each variable
    grids = {var: np.linspace(bounds[var][0], bounds[var][1], 150) for var in variables}

    # Filter combinations where both the features are non-zero
    valid_combinations = [
        (x_label, y_label) for x_label, y_label in combinations
        if bounds[x_label][0] != 0 or bounds[x_label][1] != 0
        if bounds[y_label][0] != 0 or bounds[y_label][1] != 0

    ]
    removed_items =[]
    for var in variables:
        if bounds[var][0] == 0 and bounds[var][1] == 0:
            removed_items.append(var)
    print('The following combinations are removed because the column only consists of zero values: {}'.format(removed_items))
    # Handle case where all combinations are invalid
    if not valid_combinations:
        print("No valid subplots to display. Skipping plot creation.")
        return None

    num_plots = len(valid_combinations)
    num_cols = 2
    num_rows = math.ceil(num_plots / num_cols)

    fig_height = max(5, num_plots * 3.5)
    fig_size = (d.cm2inch(16), d.cm2inch(fig_height))
    fig = plt.figure(figsize=fig_size)
    plt.subplots_adjust(left=-0.05, right=0.88, bottom=0.02, top=1, wspace=-0.1, hspace=0.05)


    for i, (x_label, y_label) in enumerate(valid_combinations, 1):
        ax = fig.add_subplot(num_rows, num_cols, i, projection="3d")
        X, Y = np.meshgrid(grids[x_label], grids[y_label])

        # Create input arrays for prediction functions
        inputs = {}
        for var in variables:
            if var == x_label:
                inputs[var] = X
            elif var == y_label:
                inputs[var] = Y
            else:
                if defaults_dict == None:
                    inputs[var] = np.full_like(X, np.mean(grids[var]))
                else:
                    inputs[var] = np.full_like(X, defaults_dict[var])

        Z = prediction_func(**inputs)
        surf = ax.plot_surface(X, Y, Z, cmap="cool", alpha=0.5)
        ax.set_box_aspect([1, 1, 0.6])
        ax.margins(x=0, y=0)
        ax.set_xlabel(x_label.replace('__', '\n'), fontsize=7, labelpad=-6)
        ax.set_ylabel(y_label.replace('__', '\n'), fontsize=7, labelpad=-6)
        ax.set_zlabel("Prediction", fontsize=7, labelpad=-7)
        ax.set_zlabel("Prediction", labelpad=-7)
        ax.tick_params(axis="x", which="major", pad=-5)
        ax.view_init(elev=30, azim=120)
        ax.tick_params(axis="y", pad=-3)
        ax.tick_params(axis="z", pad=-3)
        plt.setp(ax.get_yticklabels(), fontsize=7)
        plt.setp(ax.get_xticklabels(), fontsize=7)
        plt.setp(ax.get_zticklabels(), fontsize=7)


    # Add colorbars and label them
    cbar_ax = fig.add_axes([0.9, 0.35, 0.02, 0.3])
    cbar = fig.colorbar(surf, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label("Regressor", loc="center", fontsize=7)

    return fig


def prediction_func_4_regressor(regressor, rename_dict: dict = None):
    """
    Create a prediction function for a regressor as the regressor takes a DataFrame as input.
    """

    def pred_func(**kwargs):
        features = regressor.metadata["features_ordered"]
        if rename_dict is not None:
            features = [rename_dict[feature] for feature in features]

        # Determine the shape of the output
        shape = next(
            arr.shape for arr in kwargs.values() if isinstance(arr, np.ndarray)
        )

        # Prepare input saved_plots
        input_data = pd.DataFrame(
            {feature: np.ravel(kwargs[feature]) for feature in features}
        )

        # Make prediction
        prediction = regressor.predict(input_data)

        # Reshape the prediction to match the input shape
        return prediction.reshape(shape)

    return pred_func

def plot_carpets_with_buckets(
    model_config,
    bounds=None,
    combinations=None,
    defaults_dict=None,
    path_to_regressor=None,
    num_buckets=5):

    bucket_colors = [
        black,
        blue,
        red,
        green,
        dark_red]

    target = model_config["name_of_target"]
    # Load regressor
    if path_to_regressor is None:
        path_to_regressor = return_best_model(return_results_dir_model_tuning(model_config['name_of_raw_data'],
                                                                              model_config['name_of_data_tuning_experiment'],
                                                                              model_config[ 'name_of_model_tuning_experiment']))
    regressor = ModelFactory.load_model(path_to_regressor)
    # Do not use the input data if user provides bounds and default dictionary
    if bounds is not None and defaults_dict is not None:
        variables = regressor.metadata["features_ordered"]

    # Load the input data and fetch column names as well as bounds from it
    else:
        data_path = model_config['abs_path_to_data']
        data = load_data(data_path)
        # Fetch time column from dataset
        time_column = next((col for col in data.columns if pd.api.types.is_datetime64_any_dtype(data[col])), None)
        # Set time column as index
        if time_column:
            data.set_index(time_column, inplace=True)
        x_grid = data.drop(target, axis=1)
        variables = list(x_grid.columns)

    # Define bounds
    if bounds is None:
        bounds = {}
        for var in variables:
            if var in x_grid.columns:
                bounds[var] = [x_grid[var].min(), x_grid[var].max()]

    # Define default values
    if defaults_dict is None:
        # Use mean value as default
        defaults_dict = {var: x_grid[var].mean() for var in variables}

    # Create combinations
    if combinations is None:
        combinations = [
            (v1, v2) for i, v1 in enumerate(variables) for v2 in variables[i + 1:]
        ]

    prediction_func= prediction_func_4_regressor(regressor)

    # Create a grid for each variable
    grids = {var: np.linspace(bounds[var][0], bounds[var][1], 150) for var in variables}

    # Filter combinations where both the features are non-zero
    valid_combinations = [
        (x_label, y_label) for x_label, y_label in combinations
        if bounds[x_label][0] != 0 or bounds[x_label][1] != 0
        if bounds[y_label][0] != 0 or bounds[y_label][1] != 0

    ]
    removed_items =[]
    for var in variables:
        if bounds[var][0] == 0 and bounds[var][1] == 0:
            removed_items.append(var)
    print('The following combinations are removed because the column only consists of zero values: {}'.format(removed_items))
    # Handle case where all combinations are invalid
    if not valid_combinations:
        print("No valid subplots to display. Skipping plot creation.")
        return None

    num_plots = len(valid_combinations)
    num_cols = 2
    num_rows = math.ceil(num_plots / num_cols)

    fig_height = max(5, num_plots * 3.5)
    fig_size = (d.cm2inch(16), d.cm2inch(fig_height))
    fig = plt.figure(figsize=fig_size)
    plt.subplots_adjust(left=-0.05, right=0.88, bottom=0.05, top=1, wspace=-0.1, hspace=0.05)

    for i, (x_label, y_label) in enumerate(valid_combinations, 1):
        ax = fig.add_subplot(num_rows, num_cols, i, projection="3d")
        X, Y = np.meshgrid(grids[x_label], grids[y_label])

        # Prepare inputs for surface prediction
        inputs_surface = {}
        for var in variables:
            if var == x_label:
                inputs_surface[var] = X
            elif var == y_label:
                inputs_surface[var] = Y
            else:
                inputs_surface[var] = np.full_like(X, defaults_dict[var])

        # Create predictions based on the 2 combination values, keeping the other features fixed
        Z_surface = prediction_func(**inputs_surface)


        # Subsample for scatter points for remaining features
        all_indices = np.arange(X.size)
        n_scatter_points = int(0.005 * X.size)
        chosen_indices = np.random.choice(all_indices, n_scatter_points, replace=False)
        chosen_rows, chosen_cols = np.unravel_index(chosen_indices, X.shape)
        X_subset = X[chosen_rows, chosen_cols]
        Y_subset = Y[chosen_rows, chosen_cols]
        Z_surface_subset = Z_surface[chosen_rows, chosen_cols]

        # Prediction surface
        surf = ax.plot_surface(X, Y, Z_surface, cmap="cool", alpha=0.5, zorder=2)

        other_features = [f for f in variables if f not in (x_label, y_label)]
        bucket_edges = {f: np.linspace(bounds[f][0], bounds[f][1], num_buckets + 1) for f in other_features}

        for b in range(num_buckets):
            inputs_bucket = {
                x_label: X_subset,
                y_label: Y_subset
            }
            for f in other_features:
                lo = bucket_edges[f][b]
                hi = bucket_edges[f][b + 1]
                inputs_bucket[f] = np.random.uniform(lo, hi, size=X_subset.shape)

            # Create predictions based on buckets features
            Z_bucket = prediction_func(**inputs_bucket)

            Z_bucket_above = np.where(Z_bucket > Z_surface_subset, Z_bucket, np.nan)
            Z_bucket_below = np.where(Z_bucket <= Z_surface_subset, Z_bucket, np.nan)

            color = bucket_colors[b]

            # Lower surface
            ax.scatter(X_subset,Y_subset,Z_bucket_below,color=color,s=3,alpha=0.8,zorder=1)

            # Upper surface
            ax.scatter(X_subset,Y_subset,Z_bucket_above,color=color,s=3,alpha=0.8,zorder=3)

        ax.set_box_aspect([1, 1, 0.6])
        ax.margins(x=0, y=0)
        ax.set_xlabel(x_label.replace('__', '\n'), fontsize=7, labelpad=-6)
        ax.set_ylabel(y_label.replace('__', '\n'), fontsize=7, labelpad=-6)
        ax.set_zlabel("Prediction", fontsize=7, labelpad=-7)
        ax.set_zlabel("Prediction", labelpad=-7)
        ax.tick_params(axis="x", which="major", pad=-5)
        ax.view_init(elev=30, azim=120)
        ax.tick_params(axis="y", pad=-3)
        ax.tick_params(axis="z", pad=-3)
        plt.setp(ax.get_yticklabels(), fontsize=7)
        plt.setp(ax.get_xticklabels(), fontsize=7)
        plt.setp(ax.get_zticklabels(), fontsize=7)


    # Add legend and colorbar
    legend_entries = []
    for b in range(num_buckets):
        bucket_ranges = []
        for f in other_features:
            lo = bucket_edges[f][b]
            hi = bucket_edges[f][b + 1]
            bucket_ranges.append(f"{f}[{lo:.1f}, {hi:.1f}]")
        label = "\n".join(bucket_ranges)
        legend_entries.append((bucket_colors[b], label))

    handles = [Patch(facecolor=color, label=label) for color, label in legend_entries]
    fig.legend(
        handles=handles,
        loc='lower center',
        ncol=3,
        fontsize=6,
        frameon=False,
        handleheight=1.2)
    cbar_ax = fig.add_axes([0.9, 0.35, 0.02, 0.3])
    cbar = fig.colorbar(surf, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label("Regressor", loc="center", fontsize=7)

    return fig

#in order to see the interactivate plot in browser, use this:
 # import plotly.io as pio
 #    pio.renderers.default = "browser"
 #    plt.show()
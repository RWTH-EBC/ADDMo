import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from addmo.util import plotting as d
from pathlib import Path

from addmo.util.definitions import  return_results_dir_model_tuning, return_best_model
from addmo.s3_model_tuning.models.model_factory import ModelFactory
from addmo.util.load_save import load_data
import matplotlib.pyplot as plt
from itertools import product
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import ListedColormap
from addmo.util.plotting import *
import matplotlib.colors as colors

from matplotlib.patches import Patch


def plot_carpets(model_config, regressor, pred_func_1, pred_func_2=None,  bounds= None, combinations=None, defaults_dict=None):
    """
    Create 3D surface model_plots for prediction function.
    Note:
    pred_func_1: the regressor function
    pred_func_2: the system/measurement data
    """

    target = model_config["name_of_target"]

    # Do not use the input data if user provides bounds and default dictionary
    if bounds is not None and defaults_dict is not None:
        variables = regressor.metadata["features_ordered"]

    # Load the input data and fetch column names as well as bounds from it
    else:
        data_path = model_config['abs_path_to_data']
        data = load_data(data_path)
        measurements_data = data.drop(target, axis=1)
        variables = list(measurements_data.columns)

    # Define bounds
    if bounds is None:
        bounds = {}
        for var in variables:
            if var in measurements_data.columns:
                bounds[var] = [measurements_data[var].min(), measurements_data[var].max()]

    # Define default values
    if defaults_dict is None:
        # Use mean value as default
        defaults_dict = {var: measurements_data[var].mean() for var in variables}

    # Create combinations
    if combinations is None:
        combinations = [
            (v1, v2) for i, v1 in enumerate(variables) for v2 in variables[i + 1:]
        ]


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

        Z1 = pred_func_1(**inputs)
        surf1_cmap = "winter"
        surf2_cmap = "autumn"
        if pred_func_2 is None:
            surf1 = ax.plot_surface(X, Y, Z1, cmap=surf1_cmap, alpha=0.5)
        if pred_func_2 is not None:
            Z2 = pred_func_2(**inputs)

            # Create a common normalization for consistent coloring
            norm1 = colors.Normalize(vmin=np.nanmin(Z1), vmax=np.nanmax(Z1))
            norm2 = colors.Normalize(vmin=np.nanmin(Z2), vmax=np.nanmax(Z2))

            Z1_greater = np.where(Z1 >= Z2, Z1, np.nan)
            Z1_smaller = np.where(Z1 <= Z2, Z1, np.nan)
            Z2_greater = np.where(Z2 >= Z1, Z2, np.nan)
            Z2_smaller = np.where(Z2 <= Z1, Z2, np.nan)

            # surface plots in correct order and normalization
            surf1 = ax.plot_surface(X, Y, Z1, cmap=surf1_cmap, visible=False, norm=norm1)
            surf2 = ax.plot_surface(X, Y, Z2, cmap=surf2_cmap, visible=False, norm=norm2)
            surf2_smaller = ax.plot_surface(X, Y, Z2_smaller, cmap=surf2_cmap, alpha=0.5, norm=norm2)
            surf1_smaller = ax.plot_surface(X, Y, Z1_smaller, cmap=surf1_cmap, alpha=0.5, norm=norm1)
            surf2_greater = ax.plot_surface(X, Y, Z2_greater, cmap=surf2_cmap, alpha=0.5, norm=norm2)
            surf1_greater = ax.plot_surface(X, Y, Z1_greater, cmap=surf1_cmap, alpha=0.5, norm=norm1)

        # Add this line to reverse the axis direction
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
    if pred_func_2 is not None:
        cbar_ax1 = fig.add_axes([0.9, 0.35, 0.02, 0.3])
        cbar1 = fig.colorbar(surf1, cax=cbar_ax1)
        cbar1.set_label("Regressor")
        cbar1.set_ticks([])
        cbar1.set_ticklabels([])

        cbar_ax2 = fig.add_axes([0.9, 0.05, 0.02, 0.3])
        cbar2 = fig.colorbar(surf2, cax=cbar_ax2)
        cbar2.set_label("System")
        cbar2.set_ticks([])
        cbar2.set_ticklabels([])  # Remove tick label

    else:
        cbar_ax1 = fig.add_axes([0.9, 0.35, 0.02, 0.3])
        cbar1 = fig.colorbar(surf1, cax=cbar_ax1)
        cbar1.set_label("Regressor")
        cbar1.set_ticks([])
        cbar1.set_ticklabels([])

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
    regressor,
    pred_func_1,
    pred_func_2=None,
    bounds=None,
    combinations=None,
    defaults_dict=None,
    num_buckets=4):


    target = model_config["name_of_target"]

    # Do not use the input data if user provides bounds and default dictionary
    if bounds is not None and defaults_dict is not None:
        variables = regressor.metadata["features_ordered"]

    # Load the input data and fetch column names as well as bounds from it
    else:
        data_path = model_config['abs_path_to_data'] #Todo: is that generalized? Did you add that variable?
        data = load_data(data_path)
        measurements_data = data.drop(target, axis=1)
        variables = list(measurements_data.columns)

    # Define bounds
    if bounds is None:
        bounds = {}
        for var in variables:
            if var in measurements_data.columns:
                bounds[var] = [measurements_data[var].min(), measurements_data[var].max()]
    # Define default values
    if defaults_dict is None:
        defaults_dict = {}
        for var in variables:
            unique_vals = measurements_data[var].dropna().unique()
            if len(unique_vals) <= 3 and all(val in [0, 1] for val in unique_vals):  # binary feature
                defaults_dict[var] = measurements_data[var].mode().iloc[0] # take the mode of columns for binary features
            else:
                defaults_dict[var] = measurements_data[var].mean()

    # Create data buckets based on num of buckets:
    bucket_size = { var: ((measurements_data[var].max()- measurements_data[var].min() )/ num_buckets) for var in variables }

    bucket = {
        var: (defaults_dict[var] - (bucket_size[var]/2), defaults_dict[var] + (bucket_size[var]/2))
        for var in variables
    }

    # Create combinations
    if combinations is None:
        combinations = [
            (v1, v2) for i, v1 in enumerate(variables) for v2 in variables[i + 1:]
        ]


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
        Z1 = pred_func_1(**inputs_surface)
        surf1_cmap = "winter"
        surf2_cmap = "autumn"

        if pred_func_2 is not None:
            Z2 = pred_func_2(**inputs_surface)

            # Create a common normalization for consistent coloring
            norm1 = colors.Normalize(vmin=np.nanmin(Z1), vmax=np.nanmax(Z1))
            norm2 = colors.Normalize(vmin=np.nanmin(Z2), vmax=np.nanmax(Z2))

            Z1_greater = np.where(Z1 >= Z2, Z1, np.nan)
            Z1_smaller = np.where(Z1 <= Z2, Z1, np.nan)
            Z2_greater = np.where(Z2 >= Z1, Z2, np.nan)
            Z2_smaller = np.where(Z2 <= Z1, Z2, np.nan)

            # surface plots in correct order and normalization
            surf1 = ax.plot_surface(X, Y, Z1, cmap=surf1_cmap, visible=False, norm=norm1)
            surf2 = ax.plot_surface(X, Y, Z2, cmap=surf2_cmap, visible=False, norm=norm2)
            surf2_smaller = ax.plot_surface(X, Y, Z2_smaller, cmap=surf2_cmap, alpha=0.5, norm=norm2)
            surf1_smaller = ax.plot_surface(X, Y, Z1_smaller, cmap=surf1_cmap, alpha=0.5, norm=norm1)
            surf2_greater = ax.plot_surface(X, Y, Z2_greater, cmap=surf2_cmap, alpha=0.5, norm=norm2)
            surf1_greater = ax.plot_surface(X, Y, Z1_greater, cmap=surf1_cmap, alpha=0.5, norm=norm1)

        if pred_func_2 is None:
            norm1 = colors.Normalize(vmin=np.nanmin(Z1), vmax=np.nanmax(Z1))
            # norm2 = colors.Normalize(vmin=np.nanmin(Z2), vmax=np.nanmax(Z2))
            # filter real data points which belongs to the default dict bucket of the remaining combinations:
            other_features = [f for f in variables if f not in (x_label, y_label)]
            mask = pd.Series(True, index=measurements_data.index) # for filtering out rows which we don't want
            for f in other_features:
                lower, upper = bucket[f]
                # returns value true for the index if it falls within the range,
                # so iteratively removes indices for features which don't fall in bucket
                mask &= measurements_data[f].between(lower, upper)

            # Get filtered real data
            real_x = measurements_data.loc[mask, x_label].to_numpy()
            real_y = measurements_data.loc[mask, y_label].to_numpy()
            real_target = data.loc[mask, target].to_numpy()

            # Step 2: Prepare the grid
            x_grid = grids[x_label]
            y_grid = grids[y_label]

            # Step 3: Classify each point based on Z1 prediction from grid
            above_x, above_y, above_z = [], [], []
            below_x, below_y, below_z = [], [], []

            # Map (x_val, y_val) to Z1 prediction from the grid
            for x_val, y_val, target_val in zip(real_x, real_y, real_target):
                xi = np.abs(x_grid - x_val).argmin()
                yi = np.abs(y_grid - y_val).argmin()
                pred_val = Z1[yi, xi]  # note: row index = y, col index = x

                if target_val > pred_val:
                    above_x.append(x_val)
                    above_y.append(y_val)
                    above_z.append(target_val)
                else:
                    below_x.append(x_val)
                    below_y.append(y_val)
                    below_z.append(target_val)

            # Step 4: Split the surface Z1 into below and above parts
            Z1_below = np.where(Z1 < real_target.mean(), Z1, np.nan)  # rough approx
            Z1_above = np.where(Z1 >= real_target.mean(), Z1, np.nan)

            # ⚠️ Optional: instead of `mean`, you could make a mask grid from real_target projection

            # Step 5: Plot in order to maintain layering
            # → first the surface *below* the real data
            surf1_smaller = ax.plot_surface(X, Y, Z1_below, cmap=surf1_cmap, alpha=0.5, norm=norm1)

            # → then the real data that lies *below* the surface
            scatter= ax.scatter(below_x, below_y, below_z, color="red", label="Real < Pred", depthshade=False)
            surf1_greater = ax.plot_surface(X, Y, Z1_above, cmap=surf1_cmap, alpha=0.5, norm=norm1)

            # → then the real data that lies *above* the surface
            ax.scatter(above_x, above_y, above_z, color="green", label="Real > Pred", depthshade=False)

            # → finally the surface *above* the real data



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

    cbar_ax1 = fig.add_axes([0.92, 0.55, 0.02, 0.3])
    cbar1 = fig.colorbar(surf1_smaller, cax=cbar_ax1)
    cbar1.set_label("Regressor", fontsize=7)
    cbar1.set_ticks([])
    cbar1.set_ticklabels([])

    cbar_ax2 = fig.add_axes([0.92, 0.05, 0.02, 0.3])
    cbar2 = fig.colorbar(scatter, cax=cbar_ax2)
    cbar2.set_label("Measurement Data", fontsize=7)
    cbar2.set_ticks([])
    cbar2.set_ticklabels([])

    return fig

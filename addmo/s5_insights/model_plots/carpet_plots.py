import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from addmo.util import plotting_utils as d
from pathlib import Path

from addmo.util.definitions import  return_results_dir_model_tuning, return_best_model
from addmo.s3_model_tuning.models.model_factory import ModelFactory
from addmo.util.load_save import load_data
import matplotlib.pyplot as plt
from itertools import product
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import ListedColormap
from addmo.util.plotting_utils import *
import matplotlib.colors as colors

from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap

def truncate_colormap(cmap_name, min_val=0.1, max_val=0.9, n=256):
    """
    Truncate a colormap to exclude the extreme ends (e.g., near-white tips).
    """
    cmap = cm.get_cmap(cmap_name, n)
    new_colors = cmap(np.linspace(min_val, max_val, n))
    return LinearSegmentedColormap.from_list(f"{cmap_name}_trunc", new_colors)


def plot_carpets(variables, measurements_data, regressor_func, system_func=None,  bounds= None, combinations=None, defaults_dict=None):
    """
    Create 3D surface model_plots for prediction function.
    Note:
    regressor_func: the regressor function
    system_func: the system/measurement data
    """

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

        Z1 = regressor_func(**inputs)
        surf1_cmap = "winter"
        surf2_cmap = "autumn"
        if system_func is None:
            surf1 = ax.plot_surface(X, Y, Z1, cmap=surf1_cmap, alpha=0.5)
        if system_func is not None:
            Z2 = system_func(**inputs)

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
    if regressor_func is not None:
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

def plot_carpets_with_buckets(variables, measurements_data, target_values, regressor_func , bounds=None , combinations=None , defaults_dict=None ,num_buckets=4):

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
        ax = fig.add_subplot(num_rows, num_cols, i, projection="3d",computed_zorder=False)
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
        Z1 = regressor_func(**inputs_surface)
        surf1_cmap = "winter"
        cmap_below = truncate_colormap("YlGn_r", min_val=0, max_val=0.9)
        cmap_above = truncate_colormap("YlOrBr", min_val=0.15, max_val=1)
        #TODO: use this colormap incase of no truncation and set the background to darker grey
        # cmap_below="YlGn_r"
        # cmap_above="YlOrBr"
        # for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        #     axis.pane.set_facecolor(light_grey)

        norm1 = colors.Normalize(vmin=np.nanmin(Z1), vmax=np.nanmax(Z1))

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
        real_target = target_values.loc[mask].values.flatten()

        inputs_meas = {}
        for var in variables:
            if var == x_label:
                inputs_meas[var] = real_x
            elif var == y_label:
                inputs_meas[var] = real_y
            else:
                inputs_meas[var] = np.full_like(real_x, defaults_dict[var])
        pred_at_pts = regressor_func(**inputs_meas)

        residual = real_target - pred_at_pts
        below_mask = residual < 0
        above_mask = residual >= 0

        norm_below = colors.Normalize(vmin=residual[below_mask].min(), vmax=0)
        norm_above = colors.Normalize(vmin=0, vmax=residual[above_mask].max())
        scatter1= ax.scatter(
            real_x[below_mask], real_y[below_mask], real_target[below_mask],
            c=residual[below_mask],
            cmap=cmap_below,
            norm=norm_below,
            alpha=1, s=2, depthshade= False
        )

        surf1 = ax.plot_surface(X, Y, Z1, cmap=surf1_cmap, alpha=0.4, norm=norm1)

        scatter2 =ax.scatter(
            real_x[above_mask], real_y[above_mask], real_target[above_mask],
            c=residual[above_mask],
            cmap=cmap_above,
            norm=norm_above,
            alpha=1, s=2, depthshade=False
        )

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


    cax1 = fig.add_axes([0.92, 0.525, 0.02, 0.4])
    cb1 = fig.colorbar(surf1, cax=cax1)
    cb1.set_label("Regressor", fontsize=7)
    cb1.set_ticks([])
    cb1.set_ticklabels([])

    # for 2 colormaps:
    cax_below = fig.add_axes([0.92, 0.05, 0.02, 0.2])
    cb_below = fig.colorbar(scatter1, cax=cax_below)
    cb_below.set_label("Negative Prediction Error", fontsize=7)
    cb_below.set_ticks([])
    cb_below.set_ticklabels([])

    cax_above = fig.add_axes([0.92, 0.25, 0.02, 0.2])
    cb_above = fig.colorbar(scatter2, cax=cax_above)
    cb_above.set_label("Positive Prediction Error", fontsize=7)
    cb_above.set_ticks([])
    cb_above.set_ticklabels([])
    fig.text(0.972,0.25,"Measurement Data",rotation=90,va="center",ha="left",fontsize=7)


    return fig

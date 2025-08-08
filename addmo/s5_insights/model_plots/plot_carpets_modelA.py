import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from utilities import definitions as d
from addmo.s3_model_tuning.models.model_factory import ModelFactory
from aixtra.system_simulations import system_simulations as sys_sim


def plot_system_carpets(bounds, prediction_func_1, prediction_func_2=None, combinations=None, defaults_dict=None,):
    """
    Create comparison 3D surface plots for one or two prediction functions.
    """

    variables = list(bounds.keys())
    if combinations is None:
        combinations = [
            (v1, v2) for i, v1 in enumerate(variables) for v2 in variables[i + 1 :]
        ]

    # Create a grid for each variable
    grids = {var: np.linspace(bounds[var][0], bounds[var][1], 150) for var in variables}

    # Create the figure
    fig_size = (d.cm2inch(15.5), d.cm2inch(18)) # Adjusted figure size
    d.set_params()
    fig = plt.figure(figsize=fig_size)


    # Use gridspec for more control over subplot layout
    gs = fig.add_gridspec(
        3,
        3,
        left=0.0,
        right=1,
        bottom=0.01,
        top=1.02,
        wspace=0,
        hspace=0,
        width_ratios=[1, 1, 0.2],
    )


    latex_labels = d.meas_label

    for i, (x_label, y_label) in enumerate(combinations):
        ax = fig.add_subplot(gs[i // 2, i % 2], projection="3d")

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

        Z1 = prediction_func_1(**inputs)
        surf1_cmap = "autumn"
        surf2_cmap = "winter"
        if prediction_func_2 is None:
            surf1 = ax.plot_surface(X, Y, Z1, cmap=surf1_cmap, alpha=0.5)
        if prediction_func_2 is not None:
            Z2 = prediction_func_2(**inputs)

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
        # ax.set_xlim(ax.get_xlim()[::-1])
        ax.set_ylim(ax.get_ylim()[::-1])

        ax.set_xlabel(latex_labels.get(x_label, x_label), labelpad=-7)
        ax.set_ylabel(latex_labels.get(y_label, y_label), labelpad=-5)
        ax.set_zlabel("Prediction", labelpad=-7)
        ax.tick_params(axis="x", which="major", pad=-5)
        ax.tick_params(axis="y", pad=-3)
        ax.tick_params(axis="z", pad=-3)
        ax.view_init(elev=20, azim=120)

    # Add colorbars and label them
    if prediction_func_2 is not None:
        cbar_ax1 = fig.add_axes([0.94, 0.525, 0.02, 0.425])  # Top half
        cbar1 = fig.colorbar(surf1, cax=cbar_ax1)
        cbar1.set_label("System")
        cbar1.set_ticks([])  # Remove ticks
        cbar1.set_ticklabels([])  # Remove tick labels

        cbar_ax2 = fig.add_axes([0.94, 0.05, 0.02, 0.425])  # Bottom half
        cbar2 = fig.colorbar(surf2, cax=cbar_ax2)
        cbar2.set_label("Regressor")
        cbar2.set_ticks([])  # Remove ticks
        cbar2.set_ticklabels([])  # Remove tick labels

    else:
        cbar_ax1 = fig.add_axes([0.94, 0.3, 0.02, 0.4])
        cbar1 = fig.colorbar(surf1, cax=cbar_ax1)
        cbar1.set_label("System")
        cbar1.set_ticks([])  # Remove ticks
        cbar1.set_ticklabels([])  # Remove tick labels

    return plt


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

def execute(plot_name, plot_dir, path_to_regressor, save):

    regressor = ModelFactory.load_model(path_to_regressor)

    # Define bounds
    bounds = {
        "t_amb": [273.15 - 10, 273.15 + 20],
        "rad_dir": [0, 800],
        "u_hp": [0, 1],
        "t_room": [273.15 + 15, 273.15 + 26],
    }

    # Create all possible combinations of variables
    combinations = [
        ("u_hp", "t_amb"),
        ("u_hp", "t_room"),
        ("u_hp", "rad_dir"),
        ("t_amb", "t_room"),
        ("t_amb", "rad_dir"),
        ("rad_dir", "t_room"),
    ]

    defaults_dict = {"t_amb": 273.15, "rad_dir": 0, "u_hp": 0.5, "t_room": 273.15 + 20}

    # Create and show the plot
    plt = plot_system_carpets(
        bounds,
        sys_sim.bestest900_ODE_VL_COPcorr,
        prediction_func_4_regressor(regressor),
        combinations=combinations,
        defaults_dict=defaults_dict,
    )

    if save:
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, plot_name)
        d.save_pdf(plt, plot_path)
    else:
        plt.show()

if __name__ == "__main__":
    plot_name = "res+_modelA_carpet"
    plot_dir = os.path.join(d.results_dir(), "carpet_plots")

    pre_dir = r"R:\_Dissertationen\mre\Diss\08_Data_Plots_Analysis\0_ADDMo_TrueValidityVSExtrapolationCovargeScores\8_bes_VLCOPcorr_steady_NovDez\fullann"
    pre_dir_2 = r"8_bes_VLCOPcorr_steady_" +"vital-sweep-15"
    dir = os.path.join(pre_dir, pre_dir_2)

    path_to_regressor = os.path.join(dir, 'regressors', 'regressor.keras') #
    execute(plot_name, plot_dir, path_to_regressor, True)


import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

from addmo.util.definitions import results_dir
from addmo.s3_model_tuning.models.model_factory import ModelFactory
from aixtra.plotting import plot


def plot_system_carpets(
    bounds,
    prediction_func_1,
    prediction_func_2=None,
    combinations=None,
    defaults_dict=None,
):
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
    fig = plt.figure(figsize=(24, 15))
    plt.subplots_adjust(right=0.85)

    for i, (x_label, y_label) in enumerate(combinations, 1):
        ax = fig.add_subplot(2, 3, i, projection="3d")
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
        surf1 = ax.plot_surface(X, Y, Z1, cmap="cool", alpha=0.5)
        if prediction_func_2 is not None:
            Z2 = prediction_func_2(**inputs)
            surf2 = ax.plot_surface(X, Y, Z2, cmap="autumn", alpha=0.5)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel("Prediction")

    # Add colorbars and label them
    cbar_ax1 = fig.add_axes([0.87, 0.55, 0.02, 0.35])
    cbar1 = fig.colorbar(surf1, cax=cbar_ax1)
    cbar1.set_label("System")

    if prediction_func_2 is not None:
        cbar_ax2 = fig.add_axes([0.87, 0.1, 0.02, 0.35])
        cbar2 = fig.colorbar(surf2, cax=cbar_ax2)
        cbar2.set_label("Regressor")

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

        # Prepare input data
        input_data = pd.DataFrame(
            {feature: np.ravel(kwargs[feature]) for feature in features}
        )

        # Make prediction
        prediction = regressor.predict(input_data)

        # Reshape the prediction to match the input shape
        return prediction.reshape(shape)

    return pred_func


# Example usage:
if __name__ == "__main__":
    import aixtra.system_simulations.system_simulations as sys_sim

    path_to_regressor = r"D:\04_GitRepos\addmo-extra\aixtra_use_case\results\Empty\regressors\regressor.joblib"
    regressor = ModelFactory.load_model(path_to_regressor)

    # Define bounds
    bounds = {
        "t_amb": [273.15-10, 273.15+20],
        "rad_dir": [0, 800],
        "u_hp": [0, 1],
        "t_room": [273.15+15, 273.15+26],
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

    rename_dict = {
        "TDryBul": "t_amb",
        "HDirNor": "rad_dir",
        "oveHeaPumY_u": "u_hp",
        "reaTZon_y": "t_room",
    }

    defaults_dict = {"t_amb": 273.15, "rad_dir": 0, "u_hp": 0.5, "t_room": 273.15+20}

    # Create and show the plot
    plt = plot_system_carpets(
        bounds,
        sys_sim.bestest900_ODE_VL_COPcorr,
        # prediction_func_4_regressor(regressor, rename_dict),
        combinations=combinations,
        defaults_dict=defaults_dict,
    )
    plt.show()

    # plot.save_plot(
    #     plt, "carpets_system", results_dir()
    # )
    print(f"Plot saved: {results_dir()}.")

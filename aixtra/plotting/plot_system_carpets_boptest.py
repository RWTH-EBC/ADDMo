import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from addmo.s3_model_tuning.models.model_factory import ModelFactory
from aixtra.system_simulations.system_simulations import boptest_delta_T_air_physical_approximation_elcontrol
from matplotlib.colors import to_rgba

def prediction(t_amb, rad_dir, u_hp, t_room):
    return boptest_delta_T_air_physical_approximation_elcontrol(t_amb, rad_dir, u_hp, t_room)

def regressor_prediction(t_amb, rad_dir, u_hp, t_room):
    # Replace this with your actual regressor prediction function
    path_to_regressor = r"R:\_Dissertationen\mre\Diss\08_Data_Plots_Analysis\0_ADDMo_TrueValidityVSExtrapolationCovargeScores\7_ODEel_steady_NovDez\0PlottedCoverages\Beispiele\7_ODEel_steady_NovDez_LinReg_elated-sweep-1_COVERAGES\regressors\regressor.joblib"

    regressor = ModelFactory.load_model(path_to_regressor)

    # Find the shape of the array inputs
    shape = next(var.shape for var in [t_amb, rad_dir, u_hp, t_room] if isinstance(var, np.ndarray))

    def prepare_input(var):
        return (np.ravel(np.full(shape, var)) if isinstance(var, float)
                else np.ravel(var))

    input_data = pd.DataFrame({
        'TDryBul': prepare_input(t_amb),
        'HDirNor': prepare_input(rad_dir),
        'oveHeaPumY_u': prepare_input(u_hp),
        'reaTZon_y': prepare_input(t_room)
    })

    # Make prediction
    prediction = regressor.predict(input_data)

    # Reshape the prediction to match the input shape
    return prediction.reshape(shape)


# Define the bounds
bounds = {
    "t_amb": [263.15, 303.15],
    "rad_dir": [0, 1000],
    "u_hp": [0, 1],
    "t_room": [290.15, 300.15]
}

# Create a grid for each variable
t_amb = np.linspace(bounds["t_amb"][0], bounds["t_amb"][1], 50)
rad_dir = np.linspace(bounds["rad_dir"][0], bounds["rad_dir"][1], 50)
u_hp = np.linspace(bounds["u_hp"][0], bounds["u_hp"][1], 50)
t_room = np.linspace(bounds["t_room"][0], bounds["t_room"][1], 50)

# Create all possible combinations of variables
combinations = [
    (t_amb, rad_dir, "t_amb", "rad_dir"),
    (t_amb, u_hp, "t_amb", "u_hp"),
    (t_amb, t_room, "t_amb", "t_room"),
    (rad_dir, u_hp, "rad_dir", "u_hp"),
    (rad_dir, t_room, "rad_dir", "t_room"),
    (u_hp, t_room, "u_hp", "t_room")
]

# Create plots
fig = plt.figure(figsize=(20, 15))

# Get colors from colormaps for legend
viridis_color = to_rgba(plt.get_cmap('viridis')(0.5))
plasma_color = to_rgba(plt.get_cmap('plasma')(0.5))

for i, (x, y, x_label, y_label) in enumerate(combinations, 1):
    ax = fig.add_subplot(2, 3, i, projection='3d')
    X, Y = np.meshgrid(x, y)

    if x_label == "t_amb" and y_label == "rad_dir":
        Z1 = prediction(X, Y, np.mean(u_hp), np.mean(t_room))
        Z2 = regressor_prediction(X, Y, np.mean(u_hp), np.mean(t_room))
    elif x_label == "t_amb" and y_label == "u_hp":
        Z1 = prediction(X, np.mean(rad_dir), Y, np.mean(t_room))
        Z2 = regressor_prediction(X, np.mean(rad_dir), Y, np.mean(t_room))
    elif x_label == "t_amb" and y_label == "t_room":
        Z1 = prediction(X, np.mean(rad_dir), np.mean(u_hp), Y)
        Z2 = regressor_prediction(X, np.mean(rad_dir), np.mean(u_hp), Y)
    elif x_label == "rad_dir" and y_label == "u_hp":
        Z1 = prediction(np.mean(t_amb), X, Y, np.mean(t_room))
        Z2 = regressor_prediction(np.mean(t_amb), X, Y, np.mean(t_room))
    elif x_label == "rad_dir" and y_label == "t_room":
        Z1 = prediction(np.mean(t_amb), X, np.mean(u_hp), Y)
        Z2 = regressor_prediction(np.mean(t_amb), X, np.mean(u_hp), Y)
    else:  # u_hp and t_room
        Z1 = prediction(np.mean(t_amb), np.mean(rad_dir), X, Y)
        Z2 = regressor_prediction(np.mean(t_amb), np.mean(rad_dir), X, Y)

    surf1 = ax.plot_surface(X, Y, Z1, cmap='viridis', alpha=0.7)
    surf2 = ax.plot_surface(X, Y, Z2, cmap='plasma', alpha=0.7)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel('Prediction')
    ax.set_title(f'{x_label} vs {y_label}')

    # Add a color bar for each surface
    fig.colorbar(surf1, ax=ax, shrink=0.5, aspect=5, pad=0.1, label='Prediction')
    fig.colorbar(surf2, ax=ax, shrink=0.5, aspect=5, pad=0.15, label='Regressor Prediction')

    # Add a custom legend
    custom_lines = [plt.Line2D([0], [0], linestyle="none", c=viridis_color, marker='o'),
                    plt.Line2D([0], [0], linestyle="none", c=plasma_color, marker='o')]
    ax.legend(custom_lines, ['Prediction', 'Regressor Prediction'], loc='upper left')

plt.tight_layout()
plt.show()
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from addmo.s3_model_tuning.models.model_factory import ModelFactory
import aixtra.system_simulations.system_simulations as sys_sim
from matplotlib.colors import to_rgba

save_plot = False

# Replace this with your actual regressor prediction function

path_to_regressor = r"R:\_Dissertationen\mre\Diss\08_Data_Plots_Analysis\1_MPC_Boptest900_2019_defaultControl\ODEel_pid_steady\7_ODEel_steady_NovDez___MPC_Typ2D\Auswertung_CarpetPlot\2024-07-29_14-01__usual-sweep-3\regressor.keras"
regressor = ModelFactory.load_model(path_to_regressor)

def prediction(t_amb, rad_dir, u_hp, t_room):
    return sys_sim.boptest_delta_T_air_physical_approximation_elcontrol(t_amb, rad_dir, u_hp, t_room)

# def regressor_prediction(t_amb, rad_dir, u_hp, t_room):
#     # Find the shape of the array inputs
#     shape = next(var.shape for var in [t_amb, rad_dir, u_hp, t_room] if isinstance(var, np.ndarray))
#
#     def prepare_input(var):
#         return (np.ravel(np.full(shape, var)) if isinstance(var, float)
#                 else np.ravel(var))
#
#     input_data = pd.DataFrame({
#         'TDryBul': prepare_input(t_amb),
#         'HDirNor': prepare_input(rad_dir),
#         'oveHeaPumY_u': prepare_input(u_hp),
#         'reaTZon_y': prepare_input(t_room)
#     })
#
#     # Make prediction
#     prediction = regressor.predict(input_data)
#
#     # Reshape the prediction to match the input shape
#     return prediction.reshape(shape)

def regressor_prediction(t_amb, rad_dir, u_hp, t_room):
    return sys_sim.bestest900_ODE_VL(t_amb, rad_dir, u_hp, t_room)

# Define the bounds
bounds = {
    "t_amb": [263.15, 288.15],
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



variables = ["t_amb", "rad_dir", "u_hp", "t_room"]
n_vars = len(variables)

min_dict = {"t_room": np.min(t_room), "t_amb": np.min(t_amb), "rad_dir": np.min(rad_dir), "u_hp": np.min(u_hp)}
max_dict = {"t_room": np.max(t_room), "t_amb": np.max(t_amb), "rad_dir": np.max(rad_dir), "u_hp": np.max(u_hp)}
mean_dict = {"t_room": np.mean(t_room), "t_amb": np.mean(t_amb), "rad_dir": np.mean(rad_dir), "u_hp": np.mean(u_hp)}
winter_dict = {"t_room": 294.15, "t_amb": 273.15, "rad_dir": 0, "u_hp": 1}

used_dict = winter_dict

# Create plots
fig = plt.figure(figsize=(24, 15))

plt.subplots_adjust(right=0.85)

for i, (x, y, x_label, y_label) in enumerate(combinations, 1):
    ax = fig.add_subplot(2, 3, i, projection='3d')
    X, Y = np.meshgrid(x, y)

    if x_label == "t_amb" and y_label == "rad_dir":
        Z1 = prediction(X, Y, used_dict["u_hp"], used_dict["t_room"])
        Z2 = regressor_prediction(X, Y, used_dict["u_hp"], used_dict["t_room"])
    elif x_label == "t_amb" and y_label == "u_hp":
        Z1 = prediction(X, used_dict["rad_dir"], Y, used_dict["t_room"])
        Z2 = regressor_prediction(X, used_dict["rad_dir"], Y, used_dict["t_room"])
    elif x_label == "t_amb" and y_label == "t_room":
        Z1 = prediction(X, used_dict["rad_dir"], used_dict["u_hp"], Y)
        Z2 = regressor_prediction(X, used_dict["rad_dir"], used_dict["u_hp"], Y)
    elif x_label == "rad_dir" and y_label == "u_hp":
        Z1 = prediction(used_dict["t_amb"], X, Y, used_dict["t_room"])
        Z2 = regressor_prediction(used_dict["t_amb"], X, Y, used_dict["t_room"])
    elif x_label == "rad_dir" and y_label == "t_room":
        Z1 = prediction(used_dict["t_amb"], X, used_dict["u_hp"], Y)
        Z2 = regressor_prediction(used_dict["t_amb"], X, used_dict["u_hp"], Y)
    else:  # u_hp and t_room
        Z1 = prediction(used_dict["t_amb"], used_dict["rad_dir"], X, Y)
        Z2 = regressor_prediction(used_dict["t_amb"], used_dict["rad_dir"], X, Y)

    surf1 = ax.plot_surface(X, Y, Z1, cmap='viridis', alpha=0.7)
    surf2 = ax.plot_surface(X, Y, Z2, cmap='plasma', alpha=0.7)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel('Prediction')

# Add colorbars and label them
cbar_ax1 = fig.add_axes([0.87, 0.55, 0.02, 0.35])  # Moved to the right
cbar1 = fig.colorbar(surf1, cax=cbar_ax1)
cbar1.set_label('System')

cbar_ax2 = fig.add_axes([0.87, 0.1, 0.02, 0.35])  # Moved to the right
cbar2 = fig.colorbar(surf2, cax=cbar_ax2)
cbar2.set_label('Regressor')

if save_plot == True:
    # Save the plot in the same directory as the regressor
    regressor_dir = os.path.dirname(path_to_regressor)
    plot_filename = os.path.join(regressor_dir, 'carpet_comparison.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {plot_filename}")
plt.show()
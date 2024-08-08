import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from aixtra.system_simulations.system_simulations import boptest_delta_T_air_physical_approximation_elcontrol
def prediction(t_amb, rad_dir, u_hp, t_room):
    return boptest_delta_T_air_physical_approximation_elcontrol(t_amb, rad_dir, u_hp, t_room)

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

for i, (x, y, x_label, y_label) in enumerate(combinations, 1):
    ax = fig.add_subplot(2, 3, i, projection='3d')
    X, Y = np.meshgrid(x, y)

    if x_label == "t_amb" and y_label == "rad_dir":
        Z = prediction(X, Y, np.mean(u_hp), np.mean(t_room))
    elif x_label == "t_amb" and y_label == "u_hp":
        Z = prediction(X, np.mean(rad_dir), Y, np.mean(t_room))
    elif x_label == "t_amb" and y_label == "t_room":
        Z = prediction(X, np.mean(rad_dir), np.mean(u_hp), Y)
    elif x_label == "rad_dir" and y_label == "u_hp":
        Z = prediction(np.mean(t_amb), X, Y, np.mean(t_room))
    elif x_label == "rad_dir" and y_label == "t_room":
        Z = prediction(np.mean(t_amb), X, np.mean(u_hp), Y)
    else:  # u_hp and t_room
        Z = prediction(np.mean(t_amb), np.mean(rad_dir), X, Y)

    surf = ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel('Prediction')
    ax.set_title(f'{x_label} vs {y_label}')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

plt.tight_layout()
plt.show()
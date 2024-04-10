import numpy as np
import pandas as pd

import os

from extrapolation_detection.util import loading_saving

simulation_data_name = "Carnot_mid"

xy_tot: pd.DataFrame = loading_saving.read_csv(simulation_data_name, directory=None, index_col=False)

# add uncertainty to last feature (normally the target)
# uncertainty with a normal distribution with mean 0 and std dev 1
# Set the seed for reproducibility
np.random.seed(42)
# Number of data points to generate
num_data_points = xy_tot.shape[0]
# Generate random data from a normal distribution with mean 0 and std dev 1
std_1 = np.random.normal(loc=0, scale=1, size=num_data_points)

desired_mean = 0
desired_std = 0.02
std_custom = desired_mean + (std_1 * desired_std)

# Add the uncertainty to the last feature
xy_tot.iloc[:, -1] = xy_tot.iloc[:, -1] + std_custom

new_simulation_data_name = f"{simulation_data_name}_noise_m{desired_mean}_std{desired_std}"
path = os.path.join("edited", new_simulation_data_name + ".csv")


# save to folder
xy_tot.to_csv(path, sep=";", index=False, header=True, encoding="utf-8")


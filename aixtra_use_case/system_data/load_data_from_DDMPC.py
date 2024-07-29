import os
import pandas as pd

# Read the CSV file
df = pd.read_csv(r"D:\04_GitRepos\DDMPC_GitLab\Examples\BopTest_TAir_ODE\stored_data\data\training_data.csv")

# Select and rename the columns we need
df_new = df[['t_amb', 'rad_dir', 'u_hp', 't_room', 'Change(T Room)']].rename(columns={
    't_amb': 'TDryBul',
    'rad_dir': 'HDirNor',
    'u_hp': 'oveHeaPumY_u',
    't_room': 'reaTZon_y',
    'Change(T Room)': 'delta_reaTZon_y'
})

# shift the 'delta_reaTZon_y' column by one row to the top to make prediction be in the same row as the input sample
df_new['delta_reaTZon_y'] = df_new['delta_reaTZon_y'].shift(-1)

# check with simulation put to addmo:
from aixtra.system_simulations.system_simulations import boptest_delta_T_air_physical_approximation_elcontrol
system_prediction = boptest_delta_T_air_physical_approximation_elcontrol(df['TDryBul'], df['HDirNor'], df['oveHeaPumY_u'], df['reaTZon_y'])
# assert all but first row and last row cause there is nan in original data
error = abs(system_prediction[1:-1] - df['delta_reaTZon_y'][1:-1])
total_error = error.sum()

# if error small then put system_prediction into df_new to fill first and last row
if total_error < 1e-6:
    df_new['delta_reaTZon_y'] = system_prediction


# Save the new dataframe to a CSV file with semicolon separator and no index
new_simulation_data_name = f"ODEel_steady"
# path = os.path.join("edited", new_simulation_data_name + ".csv")
path = os.path.join(new_simulation_data_name + ".csv")



# save to folder
df_new.to_csv(path, sep=";", index=False, header=True, encoding="utf-8")

# df_new.to_csv('converted_data.csv', sep=';', index=False)

# Display the first few rows of the new dataframe
print(df_new.head())
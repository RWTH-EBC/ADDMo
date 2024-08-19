import os
import pandas as pd
from aixtra.system_simulations import system_simulations as sys_sim


# Read the CSV file
df = pd.read_csv(r"D:\04_GitRepos\DDMPC_GitLab\Examples\BopTest_TAir_ODE\stored_data\data\training_data.csv")

# Select and rename the columns we need
df_new = df[['t_amb', 'rad_dir', 'u_hp', 't_room', 'Change(T Room)']]

# shift the 'delta_reaTZon_y' column by one row to the top to make prediction be in the same row as the input sample
df_new['Change(T Room)'] = df_new['Change(T Room)'].shift(-1)

# check with simulation put to addmo:
system_prediction = sys_sim.bestest900_ODE_VL_COPcorr(df['t_amb'], df['rad_dir'], df['u_hp'], df['t_room'])
# assert all but first row and last row cause there is nan in original data
_df = df_new.copy()
_df["syspred"] = system_prediction
_df["error"] = abs(_df["syspred"] - _df['Change(T Room)'])
total_error = _df["error"].sum()
# error = abs(system_prediction[1:-1] - df['Change(T Room)'][1:-1])
# total_error = error.sum()

# if error small then put system_prediction into df_new to fill first and last row
if total_error < 1e-6:
    df_new['Change(T Room)'] = system_prediction
else:
    print(f"Error too large: {total_error}, not filling in system prediction")


# Save the new dataframe to a CSV file with semicolon separator and no index
new_simulation_data_name = f"bes_VLCOPcorr_random"
# path = os.path.join("edited", new_simulation_data_name + ".csv")
path = os.path.join(new_simulation_data_name + ".csv")



# save to folder
df_new.to_csv(path, sep=";", index=False, header=True, encoding="utf-8")

# df_new.to_csv('converted_data.csv', sep=';', index=False)

# Display the first few rows of the new dataframe
print(df_new.head())
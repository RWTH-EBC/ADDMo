import pandas as pd
from system_simulations import boptest_delta_T_air_physical_approximation

# Load the CSV file
df = pd.read_csv(
    r"D:\04_GitRepos\addmo-extra\extrapolation_detection\use_cases\data\Boptest_TAir_mid_reduced.csv",
    delimiter=";",
)

# Run the simulation for each row
for index, row in df.iterrows():
    t_amb = row["TDryBul"]
    TAirRoom = row["reaTZon_y"]
    rad_dir = row["HDirNor"]
    u_hp = row["oveHeaPumY_u"]
    df.loc[index, "delta_reaTZon_y"] = boptest_delta_T_air_physical_approximation(
        t_amb, TAirRoom, rad_dir, u_hp
    )

# Save the DataFrame to a new CSV file
df.to_csv(
    r"D:\04_GitRepos\addmo-extra\extrapolation_detection\use_cases\data\Boptest_TAir_mid_ODE.csv",
    index=False,
    sep=";",
)

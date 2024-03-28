import pandas as pd
from system_simulations import carnot_model

# Load the CSV file
df = pd.read_csv(
    r"D:\04_GitRepos\addmo-extra\extrapolation_detection\use_cases\data\Carnot_mid.csv",
    delimiter=";",
)

# Run the simulation for each row
for index, row in df.iterrows():
    t_amb = row["$T_{umg}$ in °C"]
    p_el = row["$P_{el}$ in kW"]
    q_heiz = row["$\dot{Q}_{heiz}$ in kW"]
    df.loc[index, "New"] = carnot_model(t_amb, p_el)

# Save the DataFrame to a new CSV file
df.to_csv(
    r"D:\04_GitRepos\addmo-extra\extrapolation_detection\use_cases\data\Boptest_TAir_mid_ODE.csv",
    index=False,
    sep=";",
)
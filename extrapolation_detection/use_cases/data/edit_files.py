import os

from extrapolation_detection.util import loading_saving

simulation_data_name = "Carnot_mid"
new_simulation_data_name = "Boptest_TAir_mid_reduced"
path = os.path.join("edited", new_simulation_data_name + ".csv")

xy_tot = loading_saving.read_csv(simulation_data_name, directory=None, index_col=False)
print("min:")
print(xy_tot.min())
print("max:")
print(xy_tot.max())

# delete features
delete_features = ["TDryBul-1", "Daily_Sin", "Daily_Cos", "Weekly_Sin", "Weekly_Cos",
                   "oveHeaPumY_u-1",
                   "oveHeaPumY_u-2", "reaTZon_y-1", "reaTZon_y-2"]
xy_tot = xy_tot.drop(columns=delete_features)


# save to folder
xy_tot.to_csv(path, sep=";", index=False, header=True, encoding="utf-8")

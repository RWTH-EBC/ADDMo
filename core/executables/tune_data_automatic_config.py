import yaml

from core.data_tuning.data_importer import load_raw_data
from core.data_tuning_optimizer import feature_construction
from core.data_tuning.data_tuning_config import DataTuningSetup

# load config for automatic data tuning aka GUI config
path_to_config = r"D:\04_GitRepos\addmo-extra\core\config_files\1_datatuning_GUI_config.yaml"
with open(path_to_config, "r") as f:
    config = yaml.safe_load(f)

# create data tuning setup
dt_config = DataTuningSetup()

# load data
df = load_raw_data(dt_config.abs_path_to_data)

# feature construction
x_created_1 = feature_construction.manual_feature_lags(dt_config, df)
x_created_2 = feature_construction.manual_target_lags(dt_config, df)
# x_created_3 = feature_construction.automatic_timeseries_target_lag_constructor(
#     dt_config, df
# # )
# x_created_4 = feature_construction.automatic_feature_lag_constructor(dt_config, df)




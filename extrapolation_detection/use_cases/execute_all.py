import os

from core.util.definitions import root_dir
from core.util.load_save import create_or_override_directory
from core.util.load_save import load_config_from_json
from core.util.load_save import save_config_to_json

from extrapolation_detection.use_cases.config.ed_experiment_config import (
    ExtrapolationExperimentConfig,
)
from extrapolation_detection.use_cases import (
    s1_split_data,
    s2_tune_ml_regressor,
    s3_regressor_error_calculation,
    s4_true_validity_domain,
    s5_tune_detector,
    s6_detector_score_calculation,
    s7_2_plotting,
)

# load config
# config_path = r"D:\04_GitRepos\addmo-extra\extrapolation_detection\use_cases\config\config.json"
# config = load_config_from_json(config_path, ExtrapolationExperimentConfig)
config = ExtrapolationExperimentConfig()
# config.model_validate_json(config_path)


# save config to json
serialized_config_safe_path = os.path.join(
    root_dir(),
    "extrapolation_detection",
    "use_cases",
    config.experiment_folder,
    "config.json",
)

save_config_to_json(config, serialized_config_safe_path)

# load the saved json back to a new config object
config2 = load_config_from_json(serialized_config_safe_path, ExtrapolationExperimentConfig)

print(config2==config)

s1_split_data.exe_split_data(config)
s2_tune_ml_regressor.exe_tune_regressor(config)
s3_regressor_error_calculation.exe_regressor_error_calculation(config)
s4_true_validity_domain.exe_true_validity_domain(config)
s5_tune_detector.exe_train_detector(config)
s6_detector_score_calculation.exe_detector_score_calculation(config)

s7_2_plotting.exe_plot_2D_all(config)

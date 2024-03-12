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
    s7_2_plotting
)

config = ExtrapolationExperimentConfig()

s1_split_data.exe_split_data(config)
s2_tune_ml_regressor.exe_tune_regressor(config)
s3_regressor_error_calculation.exe_regressor_error_calculation(config)
s4_true_validity_domain.exe_true_validity_domain(config)
s5_tune_detector.exe_train_detector(config)
s6_detector_score_calculation.exe_detector_score_calculation(config)

s7_2_plotting.exe_plot_2D_all(config)
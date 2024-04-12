import os

from core.util.definitions import root_dir
from core.util.load_save import load_config_from_json

from core.s3_model_tuning.config.model_tuning_config import ModelTunerConfig
from extrapolation_detection.use_cases.config.ed_experiment_config import ExtrapolationExperimentConfig
def no_tuning_config_id1(config: ExtrapolationExperimentConfig):
    config.config_explo_quant.explo_grid_points_per_axis = 10
    path_to_config = os.path.join(
        root_dir(),
        "core",
        "s3_model_tuning",
        "config",
        "model_tuner_config_no_tuning.json",
    )
    config.config_model_tuning = load_config_from_json(path_to_config, ModelTunerConfig)
    config.config_model_tuning.models = ["MLP_TargetTransformed"]
    config.config_model_tuning.hyperparameter_tuning_kwargs = {
        "hyperparameter_set": {
            "hidden_layer_sizes": [5],
            "activation": "relu",
            "max_iter": 2000,
        }
    }
    config.config_detector.detectors = ["KNN", "GP", "OCSVM"]
    return config

def tuning_config_id2(config: ExtrapolationExperimentConfig):
    config = no_tuning_config_id1(config)
    config.config_model_tuning.hyperparameter_tuning_type= "OptunaTuner"
    config.config_model_tuning.hyperparameter_tuning_kwargs = {"n_trials": 200}
    config.config_model_tuning.validation_score_metric = "neg_root_mean_squared_error"
    return config
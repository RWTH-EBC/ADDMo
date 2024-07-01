from tensorflow.keras.losses import MeanSquaredError
from addmo.s3_model_tuning.config.model_tuning_config import ModelTunerConfig
from aixtra_use_case.config.ed_experiment_config import ExtrapolationExperimentConfig


def no_tuning_config(config: ExtrapolationExperimentConfig) :
    config.config_explo_quant.explo_grid_points_per_axis = 10

    config.config_model_tuning.models = ["SciKerasSequential"]
    config.config_model_tuning.hyperparameter_tuning_type = "NoTuningTuner"
    config.config_model_tuning.hyperparameter_tuning_kwargs = {
        "hyperparameter_set": {
            "hidden_layer_sizes": [10],
            "loss": MeanSquaredError(),
            "epochs": 50
        }
    }
    config.config_model_tuning.validation_score_metric = "neg_root_mean_squared_error"
    config.config_model_tuning.validation_score_mechanism = "cv"
    config.config_detector.detectors = ["KNN", "GP", "OCSVM"]
    return config


def linear_regression_config(config: ExtrapolationExperimentConfig):
    config = no_tuning_config(config)
    config.config_model_tuning.models = ["ScikitLinearRegression"]
    config.config_model_tuning.hyperparameter_tuning_kwargs = {"hyperparameter_set": None}
    return config


def tuning_config(config: ExtrapolationExperimentConfig):
    config = no_tuning_config(config)
    config.config_model_tuning.hyperparameter_tuning_type = "OptunaTuner"
    config.config_model_tuning.hyperparameter_tuning_kwargs = {"n_trials": 20}
    config.config_model_tuning.validation_score_metric = "neg_root_mean_squared_error"
    return config

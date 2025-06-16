from keras.src.losses import MeanSquaredError
from aixtra_use_case.config.ed_experiment_config import ExtrapolationExperimentConfig


def no_tuning_config(config: ExtrapolationExperimentConfig) :
    config.config_explo_quant.explo_grid_points_per_axis = 10

    config.config_model_tuning.models = ["SciKerasSequential"]
    config.config_model_tuning.trainings_per_model = 3
    config.config_model_tuning.hyperparameter_tuning_type = "NoTuningTuner"
    config.config_model_tuning.hyperparameter_tuning_kwargs = {
        "hyperparameter_set": {
            "hidden_layer_sizes": [16],
            "epochs": 3000,
        }
    }
    config.config_model_tuning.validation_score_metric = "neg_root_mean_squared_error"
    config.config_model_tuning.validation_score_mechanism = "none"
    config.config_model_tuning.validation_score_splitting = "none"
    config.config_detector.detectors = ["KNN_untuned", "OCSVM_untuned", "GP_untuned"]#, "KNN", "OCSVM", "GP"]

    return config

def no_tuning_config_SVR(config: ExtrapolationExperimentConfig):
    config.config_explo_quant.explo_grid_points_per_axis = 10

    config.config_model_tuning.models = ["ScikitSVR"]
    config.config_model_tuning.trainings_per_model = 1
    config.config_model_tuning.hyperparameter_tuning_type = "NoTuningTuner"
    config.config_model_tuning.hyperparameter_tuning_kwargs = {"hyperparameter_set": {}}
    config.config_model_tuning.validation_score_metric = "neg_root_mean_squared_error"
    config.config_model_tuning.validation_score_mechanism = "none"
    config.config_model_tuning.validation_score_splitting = "none"
    config.config_detector.detectors = ["KNN_untuned", "OCSVM_untuned"]

    return config

def linear_regression_config(config: ExtrapolationExperimentConfig):
    config = no_tuning_config(config)
    config.config_model_tuning.models = ["ScikitLinearRegNoScaler"]
    config.config_model_tuning.trainings_per_model = 1
    config.config_model_tuning.hyperparameter_tuning_kwargs = {"hyperparameter_set": None}
    return config


def tuning_config(config: ExtrapolationExperimentConfig):
    config = no_tuning_config(config)
    config.config_model_tuning.trainings_per_model = 10
    config.config_model_tuning.hyperparameter_tuning_type = "OptunaTuner"
    config.config_model_tuning.validation_score_mechanism = "cv"
    config.config_model_tuning.validation_score_splitting = "KFold"
    config.config_model_tuning.hyperparameter_tuning_kwargs = {"n_trials": 50}
    config.config_model_tuning.validation_score_metric = "neg_root_mean_squared_error"
    return config

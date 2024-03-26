from core.s3_model_tuning.hyperparameter_tuning.hyperparameter_tuner import (
    OptunaTuner,
    NoTuningTuner,
    GridSearchTuner,
)
from core.s3_model_tuning.models.abstract_model import AbstractMLModel
from core.s3_model_tuning.config.model_tuning_config import ModelTuningExperimentConfig
from core.s3_model_tuning.hyperparameter_tuning.abstract_hyparam_tuner import (
    AbstractHyParamTuner,
)
from core.s3_model_tuning.scoring.abstract_scorer import ValidationScoring


class HyperparameterTunerFactory:
    """
    Factory for creating hyperparameter tuner instances.
    """

    @staticmethod
    def tuner_factory(config: ModelTuningExperimentConfig, scorer: ValidationScoring
                      ) -> AbstractHyParamTuner:
        """
        Creates a hyperparameter tuner based on the specified type.
        :param tuner_type: Type of tuner to create (e.g., "grid", "none", "optuna").
        :param model: The machine learning model for tuning.
        :param scoring: Scoring method to use.
        :return: Instance of a hyperparameter tuner.
        """

        if config.config_model_tuner.hyperparameter_tuning_type == "NoTuningTuner":
            return NoTuningTuner(config, scorer)
        elif config.config_model_tuner.hyperparameter_tuning_type == "OptunaTuner":
            return OptunaTuner(config, scorer)
        elif config.config_model_tuner.hyperparameter_tuning_type == "GridSearchTuner":
            return GridSearchTuner(config, scorer)
        else:
            raise ValueError("Unknown tuner type: {}".format(config.config_model_tuner.hyperparameter_tuning_type))

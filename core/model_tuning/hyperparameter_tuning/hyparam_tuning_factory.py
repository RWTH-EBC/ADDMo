from core.model_tuning.hyperparameter_tuning.hyperparameter_tuner import (
    OptunaTuner,
    NoTuningTuner,
    GridSearchTuner,
)
from core.model_tuning.models.abstract_model import AbstractMLModel
from core.model_tuning.config.model_tuning_config import ModelTuningSetup
from core.model_tuning.hyperparameter_tuning.abstract_hyparam_tuner import (
    AbstractHyParamTuner,
)
from core.model_tuning.scoring.abstract_scorer import ValidationScoring


class HyperparameterTunerFactory:
    """
    Factory for creating hyperparameter tuner instances.
    """

    @staticmethod
    def tuner_factory(config: ModelTuningSetup, model: AbstractMLModel, scorer: ValidationScoring
    ) -> AbstractHyParamTuner:
        """
        Creates a hyperparameter tuner based on the specified type.
        :param tuner_type: Type of tuner to create (e.g., "grid", "none", "optuna").
        :param model: The machine learning model for tuning.
        :param scoring: Scoring method to use.
        :return: Instance of a hyperparameter tuner.
        """
        if config.hyperparameter_tuning_type == "grid":
            return GridSearchTuner(config, model, scorer)
        elif config.hyperparameter_tuning_type == "none":
            return NoTuningTuner(config, model, scorer)
        elif config.hyperparameter_tuning_type == "optuna":
            return OptunaTuner(config, model, scorer)
        else:
            raise ValueError("Unknown tuner type: {}".format(config.hyperparameter_tuning_type))

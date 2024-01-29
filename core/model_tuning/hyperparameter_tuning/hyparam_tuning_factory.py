from core.model_tuning.hyperparameter_tuning.hyperparameter_tuner import OptunaTuner, \
    NoTuningTuner, GridSearchTuner
from core.model_tuning.models.abstract_model import AbstractMLModel
from core.model_tuning.hyperparameter_tuning.abstract_hyparam_tuner import AbstractHyParamTuner

class HyperparameterTunerFactory:
    """
    Factory for creating hyperparameter tuner instances.
    """

    @staticmethod
    def tuner_factory(tuner_type: str, model: AbstractMLModel, scoring: str) -> AbstractHyParamTuner:
        """
        Creates a hyperparameter tuner based on the specified type.
        :param tuner_type: Type of tuner to create (e.g., "grid", "none", "optuna").
        :param model: The machine learning model for tuning.
        :param scoring: Scoring method to use.
        :return: Instance of a hyperparameter tuner.
        """
        if tuner_type == "grid":
            return GridSearchTuner(model, scoring)
        elif tuner_type == "none":
            return NoTuningTuner(model, scoring)
        elif tuner_type == "optuna":
            return OptunaTuner(model, scoring)
        else:
            raise ValueError("Unknown tuner type: {}".format(tuner_type))

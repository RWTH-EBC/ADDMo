from core.model_tuning.scoring.abstract_scorer import ValidationScoring
from core.model_tuning.scoring.validator import CrossValidation
from core.model_tuning.config.model_tuning_config import ModelTuningSetup

class ValidatorFactory:
    """
    Creates and returns an instance of the specified scoring method.
    """

    @staticmethod
    def ValidatorFactory(config: ModelTuningSetup) -> ValidationScoring:
        if config.validation_score_mechanism == 'cv':
            return CrossValidation(config)
        # Add more conditions for other scoring methods
        else:
            raise ValueError("Unknown scoring type")

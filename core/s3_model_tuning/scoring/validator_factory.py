from core.s3_model_tuning.scoring.abstract_scorer import ValidationScoring
from core.s3_model_tuning.scoring.validator import CrossValidation
from core.s3_model_tuning.config.model_tuning_config import ModelTuningSetup

class ValidatorFactory:
    """
    Factory for creating validator instances.
    """

    @staticmethod
    def ValidatorFactory(config: ModelTuningSetup) -> ValidationScoring:
        if config.validation_score_mechanism == 'cv':
            return CrossValidation(config)
        # Add more conditions for other scoring mechanisms or get them dynamically like in the
        # other factory methods
        else:
            raise ValueError("Unknown validation scoring mechanism: "
                             f"{config.validation_score_mechanism}. "
                             "Available validation scoring mechanisms are: cv")

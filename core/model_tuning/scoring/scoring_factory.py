
from core.model_tuning.scoring.abstract_scoring import BaseScoring
from core.model_tuning.scoring.validation_scoring import ValidationScoring
from core.model_tuning.scoring.validation_scoring import CrossValidation

class ScoringFactory:
    """
    Creates and returns an instance of the specified scoring method.
    """

    @staticmethod
    def scoring_factory(splitting_type: str, scoring_metric:str) -> ValidationScoring:
        if splitting_type == 'KFold':
            return CrossValidation()

        # Add more conditions for other scoring methods
        else:
            raise ValueError("Unknown scoring type")

from core.model_tuning.scoring.regular_scoring import MSESoring, MAEScoring, R2Scoring
from core.model_tuning.scoring.abstract_scoring import AbstractScoring

class ScoringFactory:
    """
    Creates and returns an instance of the specified scoring method.
    """

    @staticmethod
    def scoring_factory(scoring_type: str) -> AbstractScoring:
        if scoring_type == 'MSE':
            return MSESoring()
        elif scoring_type == 'MAE':
            return MAEScoring()
        elif scoring_type == 'R2':
            return R2Scoring()
        # Add more conditions for other scoring methods
        else:
            raise ValueError("Unknown scoring type")

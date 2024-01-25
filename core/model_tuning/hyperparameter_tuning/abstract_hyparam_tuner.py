from core.model_tuning.models.abstract_model import AbstractMLModel
from core.model_tuning.scoring.abstract_scoring import AbstractScoring
from core.model_tuning.scoring.scoring_factory import ScoringFactory

class AbstractHyParamTuner:
    """
    Abstract class for hyperparameter tuning.
    """

    def __init__(self, model: AbstractMLModel, scoring: str = "R2"):
        self.model = model
        self.scorer: AbstractScoring = ScoringFactory.scoring_factory(scoring)

    def tune(self):
        """
        Abstract method for performing hyperparameter tuning.
        Returns the best hyperparameters found in the structure provided by the model.
        Also sets the model's hyperparameters to the best found.
        """
        raise NotImplementedError("Subclasses must implement this method.")

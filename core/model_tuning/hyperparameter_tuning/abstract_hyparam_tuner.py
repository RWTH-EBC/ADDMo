from core.model_tuning.models.abstract_model import AbstractMLModel
from core.model_tuning.config.model_tuning_config import ModelTuningSetup
from core.model_tuning.scoring.abstract_scorer import ValidationScoring


class AbstractHyParamTuner:
    """
    Abstract class for hyperparameter tuning.
    """

    def __init__(
        self,
        config: ModelTuningSetup,
        model: AbstractMLModel,
        scorer: ValidationScoring,
    ):
        self.config = config
        self.model = model
        self.scorer = scorer

    def tune(self, x_train_val, y_train_val, **kwargs):
        """
        Abstract method for performing hyperparameter tuning.
        Returns the best hyperparameters found in the structure provided by the model.
        Also sets the model's hyperparameters to the best found.
        """
        raise NotImplementedError("Subclasses must implement this method.")

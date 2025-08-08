from abc import ABC, abstractmethod
from addmo.s3_model_tuning.models.abstract_model import AbstractMLModel
from addmo.s3_model_tuning.config.model_tuning_config import ModelTuningExperimentConfig
from addmo.s3_model_tuning.scoring.abstract_scorer import ValidationScoring


class AbstractHyParamTuner(ABC):
    """
    Abstract class for hyperparameter tuning.
    """

    def __init__(
        self,
        config: ModelTuningExperimentConfig,
        scorer: ValidationScoring,
    ):
        self.config = config
        self.scorer = scorer

    @abstractmethod
    def tune(self, model:AbstractMLModel, x_train_val, y_train_val, **kwargs):
        """
        Abstract method for performing hyperparameter tuning.
        Returns the best hyperparameters found in the structure provided by the model.
        Also sets the model's hyperparameters to the best found.
        """
        raise NotImplementedError("Subclasses must implement this method.")

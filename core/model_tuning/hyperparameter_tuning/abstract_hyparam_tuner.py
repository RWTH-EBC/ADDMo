from core.model_tuning.models.abstract_model import AbstractMLModel
from core.model_tuning.model_configs.model_tuning_config import ModelTuningSetup

class AbstractHyParamTuner:
    """
    Abstract class for hyperparameter tuning.
    """

    def __init__(self, config: ModelTuningSetup, model: AbstractMLModel):
        self.model = model
        self.config = config

    def tune(self):
        """
        Abstract method for performing hyperparameter tuning.
        Returns the best hyperparameters found in the structure provided by the model.
        Also sets the model's hyperparameters to the best found.
        """
        raise NotImplementedError("Subclasses must implement this method.")

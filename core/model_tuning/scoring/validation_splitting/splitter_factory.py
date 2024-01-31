from core.model_tuning.scoring.validation_splitting.custom_splitters import (
    AbstractSplitter,
    NoSplitting,
    myKFold,
    TrialCustomSplitter,
)
from core.model_tuning.config.model_tuning_config import ModelTuningSetup


class SplitterFactory:
    """
    Factory for creating custom splitter instances for scikit-learn cross-validation.
    """

    @staticmethod
    def splitter_factory(config: ModelTuningSetup) -> AbstractSplitter:
        if config.validation_score_splitting == "no_splitting":
            return NoSplitting(config)
        elif config.validation_score_splitting == "kfold":
            return myKFold(config)
        elif config.validation_score_splitting == "trial_custom_splitter":
            return TrialCustomSplitter(config)
        else:
            raise ValueError(f"Unknown splitter type: {config.validation_score_splitting}")

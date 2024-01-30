from core.model_tuning.scoring.validation_splitting.custom_splitters import (
    AbstractSplitter,
    NoSplitting,
    myKFold,
    TrialCustomSplitter,
)


class SplitterFactory:
    """
    Factory for creating custom splitter instances for scikit-learn cross-validation.
    """

    @staticmethod
    def splitter_factory(splitter_type: str, **kwargs) -> AbstractSplitter:
        if splitter_type == "no_splitting":
            return NoSplitting(**kwargs)
        elif splitter_type == "kfold":
            return myKFold(**kwargs)
        elif splitter_type == "trial_custom_splitter":
            return TrialCustomSplitter(**kwargs)
        else:
            raise ValueError(f"Unknown splitter type: {splitter_type}")

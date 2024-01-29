from sklearn.model_selection import BaseCrossValidator
from core.model_tuning.scoring.splitter.custom_splitters import NoSplitting, myKFold, TrialCustomSplitter
class SplitterFactory:
    """
    Factory for creating custom splitter instances for scikit-learn cross-validation.
    """

    @staticmethod
    def splitter_factory(splitter_type: str, **kwargs) -> BaseCrossValidator:
        """
        Creates a custom splitter based on the specified type.
        :param splitter_type: Type of splitter to create (e.g., "no_splitting", "custom_kfold", "trial_custom_splitter").
        :param kwargs: Additional keyword arguments to pass to the splitter class constructor.
        :return: Instance of a custom splitter.
        """
        if splitter_type == "no_splitting":
            return NoSplitting(**kwargs)
        elif splitter_type == "custom_kfold":
            return myKFold(**kwargs)
        elif splitter_type == "trial_custom_splitter":
            return TrialCustomSplitter(**kwargs)
        else:
            raise ValueError(f"Unknown splitter type: {splitter_type}")

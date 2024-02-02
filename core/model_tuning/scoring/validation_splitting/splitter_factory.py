import inspect

from sklearn import model_selection

from core.model_tuning.scoring.validation_splitting import custom_splitters
from core.model_tuning.config.model_tuning_config import ModelTuningSetup
from core.model_tuning.scoring.validation_splitting.abstract_splitter import (
    AbstractSplitter,
)


class SplitterFactory:
    """
    Factory for creating custom splitter instances.
    """

    @staticmethod
    def splitter_factory(config: ModelTuningSetup) -> AbstractSplitter:
        """Get the custom splitter instance dynamically or use scikit-learn splitters."""

        # if splitter is custom
        if hasattr(custom_splitters, config.validation_score_splitting):
            custom_splitter_class = getattr(
                custom_splitters, config.validation_score_splitting
            )
            return custom_splitter_class(config)

        # if splitter is from scikit-learn
        elif hasattr(model_selection, config.validation_score_splitting):
            scikit_learn_splitter_class = getattr(
                model_selection, config.validation_score_splitting
            )
            if config.validation_score_splitting_kwargs is None:
                return scikit_learn_splitter_class()
            else:
                return scikit_learn_splitter_class(
                    **config.validation_score_splitting_kwargs
                )

        # if splitter is not found
        else:
            # get the names of all custom splitters for error message
            custom_splitter_names = [
                name
                for name, obj in inspect.getmembers(custom_splitters)
                if inspect.isclass(obj)
                and issubclass(obj, AbstractSplitter)
                and name is not "AbstractSplitter"
            ]

            raise ValueError(
                f"Unknown splitter type: {config.validation_score_splitting}. "
                f"Available custom splitter are:"
                f" {', '.join(custom_splitter_names)}. "
                f"You can also use any splitter from scikit-learn, like KFold, "
                f"PredefinedSplit, TimeSeriesSplit, etc."
            )

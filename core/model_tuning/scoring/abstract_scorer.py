from abc import ABC, abstractmethod

from core.model_tuning.config.model_tuning_config import ModelTuningSetup
from core.model_tuning.models.abstract_model import AbstractMLModel
from core.model_tuning.scoring.metrics.abstract_metric import AbstractMetric
from core.model_tuning.scoring.metrics.metric_factory import MetricFactory
from core.model_tuning.scoring.validation_splitting.splitter_factory import (
    SplitterFactory,
)


class Scoring:
    """This class is used to score the model on the test period.
    Currently I dont see any customizations that couldnt be implemented directly into the custom
    metric class. Hence, this class is not designed as abstract class."""

    def __init__(self, scoring_metric: str):
        self.metric: AbstractMetric = MetricFactory.metric_factory(scoring_metric)

    def score_test(self, model: AbstractMLModel, x, y):
        """Returns a positive float value. The higher the better.
        x and y include only test period. The model is already trained."""
        return self.metric(model, x, y)


class ValidationScoring(ABC):
    def __init__(self, config: ModelTuningSetup):
        self.metric = MetricFactory.metric_factory(
            config.validation_score_metric, config.validation_score_metric_kwargs
        )
        self.splitter = SplitterFactory.splitter_factory(config)

    @staticmethod
    @abstractmethod
    def score_validation(model: AbstractMLModel, x, y):
        """Returns a positive float value. The higher the better.
        x and y include train and evaluation period. The model will be trained on the corresponding
        train period."""
        pass

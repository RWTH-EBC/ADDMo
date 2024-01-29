from abc import ABC
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import BaseCrossValidator


from core.model_tuning.scoring.custom_splitters import NoSplitting, myKFold, \
    TrialCustomSplitter

from core.model_tuning.models.abstract_model import AbstractMLModel



class BaseScoring(ABC):

    def __init__(self, metric_name:str):
        self.metric = self.get_metric(metric_name)
    @staticmethod #todo: als factory?
    def get_metric(self, metric_name:str):
        # if metric is from scikit learn
        if metric_name in metrics.get_scorer_names():
            metric = metrics.get_scorer(metric_name)
        # if metric is custom
        else:
            # metric = custom_metric_factory(metric_name)
            pass

        return metric

    @staticmethod
    def get_splitter(splitter_type: str) -> BaseCrossValidator:
        """
        Creates a custom splitter based on the specified type.
        :param splitter_type: Type of splitter to create (e.g., "no_splitting", "custom_kfold", "trial_custom_splitter").
        :param kwargs: Additional keyword arguments to pass to the splitter class constructor.
        :return: Instance of a custom splitter.
        """
        if splitter_type == "no_splitting":
            return NoSplitting()
        elif splitter_type == "custom_kfold":
            return myKFold()
        elif splitter_type == "trial_custom_splitter":
            return TrialCustomSplitter()
        else:
            raise ValueError(f"Unknown splitter type: {splitter_type}")

class TestScoring(BaseScoring):
    def score_test(self, model: AbstractMLModel, metric_name:str, x, y):
        """ Returns a positive float value. The higher the better.
        x and y include only test period."""
        return self.metric(model, metric_name, x, y)

class ValidationScoring(BaseScoring):

    def __init__(self, metric_name:str, splitter_name:str):
        super().__init__()
        self.splitter = self.get_splitter(splitter_name)

    @staticmethod
    def score_validation(model: AbstractMLModel, metric_name:str, x, y):
        """ Returns a positive float value. The higher the better.
        x and y include train and evaluation period."""

        pass

class CrossValidation(ValidationScoring):

    def score_validation(self, model, metric_name, x, y, **cv_kwargs):
        """ Returns a positive float value. The higher the better.
        x and y include train and evaluation period.
        CV is shuffle=False by default, so the splits will be same across calls."""

        scores = cross_val_score(model, x, y, scoring=self.metric, cv=self.splitter, **cv_kwargs)
        return scores.mean()
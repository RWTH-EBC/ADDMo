from abc import ABC, abstractmethod
from sklearn import metrics


from core.model_tuning.models.abstract_model import AbstractMLModel



class BaseScoring(ABC):
    @staticmethod
    def get_metric(self, metric_name:str):
        # if metric is from scikit learn
        if metric_name in metrics.get_scorer_names():
            metric = metrics.get_scorer(metric_name)
        # if metric is custom
        else:
            # metric = custom_metric_factory(metric_name)
            pass

        return metric

    def calc_metric(self, model: AbstractMLModel, metric_name:str, x, y):
        """ Returns a positive float value. The higher the better."""
        metric_scorer = self.get_metric(metric_name)
        return metric_scorer(model, x, y)

class TestScoring(BaseScoring):
    def score_test(self, model: AbstractMLModel, metric_name:str, x, y):
        """ Returns a positive float value. The higher the better.
        x and y include only test period."""
        return self.calc_metric(model, metric_name, x, y)

class ValidationScoring(BaseScoring):
    @staticmethod
    def score_validation(model: AbstractMLModel, metric_name:str, x, y):
        """ Returns a positive float value. The higher the better.
        x and y include train and evaluation period."""
        pass
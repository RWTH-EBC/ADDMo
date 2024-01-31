from sklearn.model_selection import cross_val_score

from core.model_tuning.scoring.abstract_scorer import ValidationScoring
from core.model_tuning.models.abstract_model import AbstractMLModel

class CrossValidation(ValidationScoring):

    def score_validation(self, model:AbstractMLModel, x, y):
        """ Returns a positive float value. The higher the better.
        x and y include train and evaluation period.
        CV is shuffle=False by default, so the splits will be same across calls."""

        scores = cross_val_score(model.to_scikit_learn(), x, y, scoring=self.metric, cv=self.splitter)
        return scores.mean()

class TimeSeriesSplitting(ValidationScoring):
    pass
from sklearn.model_selection import cross_validate

from core.model_tuning.scoring.abstract_scorer import ValidationScoring
from core.model_tuning.models.abstract_model import AbstractMLModel

from core.util.experiment_logger import WandbLogger

class CrossValidation(ValidationScoring):

    def score_validation(self, model:AbstractMLModel, x, y):
        """ Returns a positive float value. The higher the better.
        x and y include train and evaluation period.
        CV is shuffle=False by default, so the splits will be same across calls."""

        info = cross_validate(model.to_scikit_learn(), x, y, scoring=self.metric,
                              cv=self.splitter, return_indices=True)

        # log the dataset splits for specific splitters which are good important to check
        if self.splitter.__class__.__name__ == "KrasserSplitter":
            splitter_indices:dict = info["indices"]
            WandbLogger(splitter_indices)


        scores = info["test_score"]
        return scores.mean()




class TimeSeriesSplitting(ValidationScoring):
    pass
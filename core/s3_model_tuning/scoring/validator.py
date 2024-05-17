from sklearn.model_selection import cross_validate

from core.s3_model_tuning.scoring.abstract_scorer import ValidationScoring
from core.s3_model_tuning.models.abstract_model import AbstractMLModel

from core.util.experiment_logger import WandbLogger


class CrossValidation(ValidationScoring):
    def score_validation(self, model: AbstractMLModel, x, y):
        """Returns a positive float value. The higher the better.
        x and y include train and evaluation period.
        CV is shuffle=False by default, so the splits will be same across calls."""


        cv_info = cross_validate(
            model.to_scikit_learn(x),
            x,
            y,
            scoring=self.metric,
            cv=self.splitter,
            return_indices=True
        )

        # log the dataset splits for specific splitters which are important to check
        # if self.splitter.__class__.__name__ == "UnivariateSplitter":
        #     splitter_indices: dict = cv_info["indices"]
        #     # convert indices to datetime indices
        #     splitter_indices["train"] = [
        #         x.iloc[splitter_indices["train"][i]].index
        #         for i in range(len(splitter_indices["train"]))
        #     ]
        #     splitter_indices["test"] = [
        #         x.iloc[splitter_indices["test"][i]].index
        #         for i in range(len(splitter_indices["test"]))
        #     ]
        #     # WandbLogger(splitter_indices)

        scores = cv_info["test_score"]
        return scores.mean()

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from core.model_tuning.scoring.abstract_scoring import ValidationScoring



class TrainTestSplit(ValidationScoring):

    def score_validation(self, model, metric_name, x, y, **kwargs):
        """
        Returns a positive float value. The higher the better.
        x and y include train and evaluation period.
        Test split size is 25 % by default
        """
        # fix random_state so the splits will be same across calls.
        if "random_state" not in kwargs:
            kwargs["random_state"] = 3

        x_train, x_test, y_train, y_test = train_test_split(x, y, **kwargs)

        model.fit(x_train, y_train)

        score = self.calc_metric(model, metric_name, x_test, y_test)
        return score
class CrossValidation(ValidationScoring):

    def score_validation(self, model, metric_name, x, y, **cv_kwargs):
        """ Returns a positive float value. The higher the better.
        x and y include train and evaluation period.
        CV is shuffle=False by default, so the splits will be same across calls."""

        scores = cross_val_score(model, x, y, **cv_kwargs)
        return scores.mean()
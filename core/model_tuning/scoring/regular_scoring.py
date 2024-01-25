# scoring_functions.py

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from core.model_tuning.scoring.abstract_scoring import AbstractScoring
from core.model_tuning.models.abstract_model import AbstractMLModel

class MSESoring(AbstractScoring):
    @staticmethod
    def score(model: AbstractMLModel, X, y):
        predictions = model.infer(X)
        mse = mean_squared_error(y, predictions)
        return -mse  # Negated to ensure a positive value

class MAEScoring(AbstractScoring):
    @staticmethod
    def score(model: AbstractMLModel, X, y):
        predictions = model.infer(X)
        mae = mean_absolute_error(y, predictions)
        return -mae  # Negated to ensure a positive value

class R2Scoring(AbstractScoring):
    @staticmethod
    def score(model: AbstractMLModel, X, y):
        predictions = model.infer(X)
        r2 = r2_score(y, predictions)
        return r2  # R2 is naturally a positive score

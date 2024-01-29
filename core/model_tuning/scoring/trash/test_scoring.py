# # scoring_functions.py
#
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
#
# from core.model_tuning.scoring.abstract_scoring import TestScoring
# from core.model_tuning.models.abstract_model import AbstractMLModel
#
# class MSESoring(TestScoring):
#     @staticmethod
#     def score_test(model: AbstractMLModel, X, y):
#         predictions = model.predict(X)
#         mse = mean_squared_error(y, predictions)
#         return -mse  # Negated to ensure a positive value
#
# class MAEScoring(TestScoring):
#     @staticmethod
#     def score_test(model: AbstractMLModel, X, y):
#         predictions = model.predict(X)
#         mae = mean_absolute_error(y, predictions)
#         return -mae  # Negated to ensure a positive value
#
# class R2Scoring(TestScoring):
#     @staticmethod
#     def score_test(model: AbstractMLModel, X, y):
#         predictions = model.predict(X)
#         r2 = r2_score(y, predictions)
#         return r2  # R2 is naturally a positive score_test


#todo: all implemented in sklearn
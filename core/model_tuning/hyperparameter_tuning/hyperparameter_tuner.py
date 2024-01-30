# Import necessary libraries
import optuna
from optuna.integration import WeightsAndBiasesCallback
import wandb
import inspect
from sklearn.model_selection import GridSearchCV, cross_validate

from core.model_tuning.hyperparameter_tuning.abstract_hyparam_tuner import AbstractHyParamTuner
from core.model_tuning.scoring.abstract_scorer import ValidationScoring
from core.model_tuning.config.model_tuning_config import ModelTuningSetup

class NoTuningTuner(AbstractHyParamTuner):
    """
    Tuner implementation for no tuning.
    """
    def tune(self, hyperparameter_set=None): #Todo: überlegen wo dann die yaml mit spezifischen
        # hyperparametern hin soll und eingeladen wird
        """
        Returns the default hyperparameters without any tuning.
        """
        if hyperparameter_set is None:
            hyperparameter_set = self.model.default_hyperparameter()

        self.model.set_params(self.model.default_hyperparameter())

        self.model.set_params()
        return hyperparameter_set

class OptunaTuner(AbstractHyParamTuner):
    def tune(self, x_train_val, y_train_val):
        """
        Perform hyperparameter tuning using Optuna.
        :param n_trials: Number of optimization trials.
        """

        # wandb_kwargs = {"project": "my-project"}
        # wandbc = WeightsAndBiasesCallback(wandb_kwargs=wandb_kwargs, as_multirun=True)
        #
        # wandb.log(
        #     {
        #         "hyperparameter_range": inspect.getsource(
        #             self.model.optuna_hyperparameter_suggest
        #         )
        #     }
        # )

        #@wandbc.track_in_wandb()
        def objective(trial):
            hyperparameters = self.model.optuna_hyperparameter_suggest(trial)
            self.model.set_params(hyperparameters)
            score = self.scorer.score_validation(self.model, x_train_val, y_train_val)
            # wandb.log({"score_test": score, "hyperparameters": hyperparameters})
            return score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.config.iterations_hyperparameter_tuning)
        # , callbacks=[wandbc])
        # wandb.finish()

        # convert optuna params to model params
        best_params = self.model.optuna_hyperparameter_suggest(study.best_trial)
        self.model.set_params(best_params)
        return best_params

class GridSearchTuner(AbstractHyParamTuner): # Todo
    """
    Tuner implementation using Grid Search.
    """

    def tune(self, cv=5):
        """
        Perform hyperparameter tuning using Grid Search.
        """
        #todo: actually i want to have CV within the scoring function

        grid_search = GridSearchCV(self.model, self.model.gridsearch_hyperparameter(),
                                   cv=cv,
                                   scoring=self.scorer)
        grid_search.fit(self.model.data, self.model.target)

        hyperparameters = grid_search.best_params_

        self.model.set_params(hyperparameters)
        return hyperparameters

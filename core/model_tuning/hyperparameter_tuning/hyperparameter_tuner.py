# Import necessary libraries
import optuna
from optuna.integration import WeightsAndBiasesCallback
import wandb
import inspect

from core.model_tuning.hyperparameter_tuning.abstract_hyparam_tuner import AbstractHyParamTuner

class NoTuningTuner(AbstractHyParamTuner):
    """
    Tuner implementation for no tuning.
    """
    def tune(self, hyperparameters=None): #Todo: überlegen wo dann die yaml mit spezifischen
        # hyperparametern hin soll und eingeladen wird
        """
        Returns the default hyperparameters without any tuning.
        """
        if hyperparameters is None:
            hyperparameters = self.model.default_hyperparameter()

        self.model.set_hyperparameters(self.model.default_hyperparameter())

        self.model.set_hyperparameters()
        return hyperparameters

class OptunaTuner(AbstractHyParamTuner):
    def tune(self, n_trials=100):
        """
        Perform hyperparameter tuning using Optuna.
        :param n_trials: Number of optimization trials.
        """

        wandb_kwargs = {"project": "my-project"}
        wandbc = WeightsAndBiasesCallback(wandb_kwargs=wandb_kwargs, as_multirun=True)

        wandb.log(
            {
                "hyperparameter_range": inspect.getsource(
                    self.model.optuna_hyperparameter_suggest
                )
            }
        )

        @wandbc.track_in_wandb()
        def objective(trial):
            hyperparameters = self.model.optuna_hyperparameter_suggest(trial)
            self.model.set_hyperparameters(hyperparameters)
            score = self.scorer()
            wandb.log({"score_test": score, "hyperparameters": hyperparameters})
            return score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, callbacks=[wandbc])
        wandb.finish()

        self.model.set_hyperparameters(study.best_params)
        return study.best_params

class GridSearchTuner(AbstractHyParamTuner):
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

        self.model.set_hyperparameters(hyperparameters)
        return hyperparameters

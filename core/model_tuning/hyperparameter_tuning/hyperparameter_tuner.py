import inspect

import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback
from sklearn.model_selection import GridSearchCV


from core.model_tuning.models.abstract_model import AbstractMLModel
from core.model_tuning.hyperparameter_tuning.abstract_hyparam_tuner import (
    AbstractHyParamTuner,
)
from core.util.experiment_logger import ExperimentLogger
from core.util.experiment_logger import WandbLogger


class NoTuningTuner(AbstractHyParamTuner):
    """
    Tuner implementation for no tuning.
    """

    def tune(self, model: AbstractMLModel, x_train_val, y_train_val, **kwargs):
        """
        Returns the default hyperparameters without any tuning.
        """

        # if no kwargs are given, use default hyperparameters
        hyperparameter_set = kwargs.get("hyperparameter_set", None)
        if hyperparameter_set is None:
            hyperparameter_set = model.default_hyperparameter()
        return hyperparameter_set


class OptunaTuner(AbstractHyParamTuner):
    def tune(self, model: AbstractMLModel, x_train_val, y_train_val, **kwargs):
        """
        Perform hyperparameter tuning using Optuna.
        """

        ExperimentLogger.log(
            {
                "hyperparameter_range": inspect.getsource(
                    model.optuna_hyperparameter_suggest
                )
            }
        )


        def objective(trial):
            hyperparameters = model.optuna_hyperparameter_suggest(trial)
            model.set_params(hyperparameters)
            score = self.scorer.score_validation(model, x_train_val, y_train_val)
            return score

        study = optuna.create_study(direction="maximize")
        study.optimize(
            objective,
            n_trials=self.config.hyperparameter_tuning_kwargs["n_trials"],
            n_jobs=1,  # The number of parallel jobs. If this argument is set to -1, the number
            # is set to CPU count. Parallel jobs may fail sequential logging to wandb.
            # callbacks=wandbc # logging to wandb, no logging if wandbc is None
        )

        # convert optuna params to model params
        best_params = model.optuna_hyperparameter_suggest(study.best_trial)
        return best_params


class GridSearchTuner(AbstractHyParamTuner):
    """
    Tuner implementation using Grid Search.
    """

    def tune(self, model: AbstractMLModel, x_train_val, y_train_val, **kwargs):
        """
        Perform hyperparameter tuning using Grid Search.
        """

        grid_search = GridSearchCV(
            model,
            model.grid_search_hyperparameter(),
            cv=self.scorer.splitter,
            scoring=self.scorer.metric,
        )
        grid_search.fit(x_train_val, y_train_val)

        best_params = grid_search.best_params_
        return best_params

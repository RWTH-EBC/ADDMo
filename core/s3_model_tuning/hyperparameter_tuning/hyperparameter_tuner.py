import inspect
import optuna
import wandb
from sklearn.model_selection import GridSearchCV
from core.s3_model_tuning.models.abstract_model import AbstractMLModel
from core.s3_model_tuning.hyperparameter_tuning.abstract_hyparam_tuner import (
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
            print("No hyperparameter set given, will use default hyperparameters.")
        return hyperparameter_set


class OptunaTuner(AbstractHyParamTuner):
    def tune(self, model: AbstractMLModel, x_train_val, y_train_val, **kwargs):
        """
        Perform hyperparameter tuning using Optuna.
        """

        def objective(trial):
            hyperparameters = model.optuna_hyperparameter_suggest(trial)
            print(hyperparameters)
            model.set_params(hyperparameters)
            score = self.scorer.score_validation(model, x_train_val, y_train_val)
            return score

        study = optuna.create_study(direction="maximize")
        study.optimize(
            objective,
            n_trials=self.config.hyperparameter_tuning_kwargs["n_trials"],
            n_jobs=-1,  # The number of parallel jobs. If this argument is set to -1, the number
            # is set to CPU count. Parallel jobs may fail sequential logging to wandb.
        )

        # logging
        self._log_optuna_study(study, model)

        # convert optuna params to model params
        best_params = model.optuna_hyperparameter_suggest(study.best_trial)
        return best_params

    @staticmethod
    def _log_optuna_study(study, model):
        ExperimentLogger.log_artifact(study, "optuna_study", art_type="pkl")
        hyperparameter_range = {"hyperparameter_range": inspect.getsource(
                    model.optuna_hyperparameter_suggest
                )
            }
        study_results = {
            "optuna_study_best_params": study.best_params,
            "optuna_study_best_validation_score": study.best_value,
        }
        plots = {
            "optuna_plot_optimization_history": optuna.visualization.plot_optimization_history(
                study
            ),
            "optuna_plot_parallel_coordinate": optuna.visualization.plot_parallel_coordinate(
                study
            ),
            "optuna_plot_contour": optuna.visualization.plot_contour(study),
            "optuna_plot_param_importances": optuna.visualization.plot_param_importances(
                study
            ),
        }
        ExperimentLogger.log({**hyperparameter_range, **study_results, **plots})


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

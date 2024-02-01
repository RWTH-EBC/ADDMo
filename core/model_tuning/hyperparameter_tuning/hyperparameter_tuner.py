import optuna
from sklearn.model_selection import GridSearchCV

from core.model_tuning.models.abstract_model import AbstractMLModel
from core.model_tuning.hyperparameter_tuning.abstract_hyparam_tuner import AbstractHyParamTuner

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
        model.set_params(model.default_hyperparameter())

        return hyperparameter_set

class OptunaTuner(AbstractHyParamTuner):
    def tune(self, model: AbstractMLModel, x_train_val, y_train_val, **kwargs):
        """
        Perform hyperparameter tuning using Optuna.
        """

        # wandb_kwargs = {"project": "my-project"}
        # wandbc = WeightsAndBiasesCallback(wandb_kwargs=wandb_kwargs, as_multirun=True)
        #
        # wandb.log(
        #     {
        #         "hyperparameter_range": inspect.getsource(
        #             model.optuna_hyperparameter_suggest
        #         )
        #     }
        # )

        #@wandbc.track_in_wandb()
        def objective(trial):
            hyperparameters = model.optuna_hyperparameter_suggest(trial)
            model.set_params(hyperparameters)
            score = self.scorer.score_validation(model, x_train_val, y_train_val)
            # wandb.log({"score_test": score, "hyperparameters": hyperparameters})
            return score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.config.hyperparameter_tuning_kwargs["n_trials"])
        # , callbacks=[wandbc])
        # wandb.finish()

        # convert optuna params to model params
        study.best_params = model.optuna_hyperparameter_suggest(study.best_trial)
        model.set_params(study.best_params)
        return study.best_params

class GridSearchTuner(AbstractHyParamTuner):
    """
    Tuner implementation using Grid Search.
    """

    def tune(self, model: AbstractMLModel, x_train_val, y_train_val, **kwargs):
        """
        Perform hyperparameter tuning using Grid Search.
        """

        grid_search = GridSearchCV(model, model.grid_search_hyperparameter(),
                                   cv=self.scorer.splitter,
                                   scoring=self.scorer.metric)
        grid_search.fit(x_train_val, y_train_val)

        best_params = grid_search.best_params_

        model.set_params(best_params)
        return best_params

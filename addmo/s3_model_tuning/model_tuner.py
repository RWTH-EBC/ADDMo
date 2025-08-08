from copy import deepcopy

from sklearn.metrics import root_mean_squared_error

from addmo.s3_model_tuning.config.model_tuning_config import ModelTunerConfig
from addmo.s3_model_tuning.models.model_factory import ModelFactory
from addmo.s3_model_tuning.hyperparameter_tuning.hyparam_tuning_factory import (
    HyperparameterTunerFactory,
)
from addmo.s3_model_tuning.scoring.validator_factory import ValidatorFactory


class ModelTuner:
    def __init__(self, config: ModelTunerConfig):
        self.config = config
        self.scorer = ValidatorFactory.ValidatorFactory(self.config)
        self.tuner = HyperparameterTunerFactory.tuner_factory(self.config, self.scorer)

    def tune_model(self, model_name: str, x_train_val, y_train_val):
        """
        Tune a model and return the best model fitted to training and validation system_data.
        """
        model = ModelFactory.model_factory(model_name)

        best_params = self.tuner.tune(
            model, x_train_val, y_train_val, **self.config.hyperparameter_tuning_kwargs
        )

        model.set_params(best_params)

        # due to refitting on each fold, the validation score must be calculated before fitting
        model.validation_score = self.scorer.score_validation(
            model, x_train_val, y_train_val
        )

        # refit the model on the whole training and validation system_data and get best model
        fitted_models = []
        for i in range(self.config.trainings_per_model):
            _model = deepcopy(model)
            _model.fit(x_train_val, y_train_val)
            y_pred = _model.predict(x_train_val)
            _model.fit_error = root_mean_squared_error(y_train_val, y_pred)
            print(f"Model training {i} fit error: {_model.fit_error}")
            fitted_models.append(_model)
        model = min(fitted_models, key=lambda x: x.fit_error)

        return model

    def tune_all_models(self, x_train_val, y_train_val):
        """
        Tune all models and return the best model fitted to training and validation system_data.
        """
        model_dict = {}
        for model_name in self.config.models:
            model = self.tune_model(model_name, x_train_val, y_train_val)
            model_dict[model_name] = model
        return model_dict

    def get_model_validation_score(self, model_dict, model_name):
        """
        Get the model validation score from the model dictionary.
        """
        return model_dict[model_name].validation_score

    def get_best_model_name(self, model_dict):
        """
        Get the best model name from the model dictionary.
        """
        best_model_name = max(
            model_dict, key=lambda x: self.get_model_validation_score(model_dict, x)
        )
        return best_model_name
    def get_model(self, model_dict, model_name):
        """
        Get the model from the model dictionary.
        """
        return model_dict[model_name]



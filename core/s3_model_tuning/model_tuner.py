from core.s3_model_tuning.config.model_tuning_config import ModelTunerConfig
from core.s3_model_tuning.models.model_factory import ModelFactory
from core.s3_model_tuning.hyperparameter_tuning.hyparam_tuning_factory import (
    HyperparameterTunerFactory,
)
from core.s3_model_tuning.scoring.validator_factory import ValidatorFactory


class ModelTuner:
    def __init__(self, config: ModelTunerConfig):
        self.config = config
        self.scorer = ValidatorFactory.ValidatorFactory(self.config)
        self.tuner = HyperparameterTunerFactory.tuner_factory(self.config, self.scorer)

    def tune_model(self, model_name: str, x_train_val, y_train_val):
        """Tune a model and return the best model fitted to training and validation data."""
        model = ModelFactory.model_factory(model_name)

        best_params = self.tuner.tune(
            model, x_train_val, y_train_val, **self.config.hyperparameter_tuning_kwargs
        )

        model.set_params(best_params)

        model.fit(x_train_val, y_train_val)

        return model

    def tune_all_models(self, x_train_val, y_train_val):
        model_dict = {}
        for model_name in self.config.models:
            model = self.tune_model(model_name, x_train_val, y_train_val)
            validation_score = self.scorer.score_validation(
                model, x_train_val, y_train_val
            )

            model_dict[model_name] = {
                "model": model,
                "validation_score": validation_score,
            }
        return model_dict

    def get_best_model_name(self, model_dict):
        best_model_name = max(
            model_dict, key=lambda x: model_dict[x]["validation_score"]
        )
        return best_model_name
    def get_model(self, model_dict, model_name):
        return model_dict[model_name]["model"]
    def get_model_validation_score(self, model_dict, model_name):
        return model_dict[model_name]["validation_score"]


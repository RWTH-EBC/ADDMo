from core.model_tuning.config.model_tuning_config import ModelTuningSetup
from core.util.experiment_logger import ExperimentLogger
from core.data_tuning.data_importer import load_data
from core.model_tuning.models.model_factory import ModelFactory
from core.model_tuning.hyperparameter_tuning.hyparam_tuning_factory import (
    HyperparameterTunerFactory,
)
from core.model_tuning.scoring.validator_factory import ValidatorFactory
from core.util.data_handling import split_target_features


class ModelTuner:
    def __init__(self, config: ModelTuningSetup):
        self.config = config
        self.xy = load_data(self.config.abs_path_to_data)

    def tune_model(self):
        model = ModelFactory.model_factory(self.config.models[0])

        scorer = ValidatorFactory.ValidatorFactory(self.config)

        tuner = HyperparameterTunerFactory.tuner_factory(self.config, model, scorer)

        x_train_val, y_train_val = split_target_features(
            self.config.name_of_target,
            self.xy
        )

        best_params = tuner.tune(x_train_val, y_train_val)

        model.set_params(best_params)

        model.fit(x_train_val, y_train_val)

        return model
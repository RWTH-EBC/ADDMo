from core.model_tuning.model_configs.model_tuning_config import ModelTuningSetup
from core.util.experiment_logger import ExperimentLogger
from core.data_tuning.data_importer import load_data
from core.model_tuning.models.model_factory import ModelFactory
from core.model_tuning.hyperparameter_tuning.hyperparameter_tuning_factory import HyParamTuningFactory
from core.model_tuning.scoring.scoring_factory import ScoringFactory


class model_tuner():
    def __init__(self, config: ModelTuningSetup, logger: ExperimentLogger):
        self.config = config
        self.logger = logger
        self.xy = load_data(self.config.abs_path_to_data)

    def _get_models(self):
        '''Returns a list of models to be tuned'''
        for model_name in self.config.model_names:
            yield ModelFactory.model_factory(model_name)
    def tune_model(self, model_name):
        model = ModelFactory.model_factory()
        scorer = ScoringFactory.scoring_factory(self.config.validation_score_splitting)
        tuner = HyParamTuningFactory.tuner_factory(self.config.tuner_type, model, self.config.scoring_metric)








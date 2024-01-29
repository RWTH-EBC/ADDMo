from core.model_tuning.models.model_factory import ModelFactory
from core.model_tuning.models.abstract_model import AbstractMLModel
from core.model_tuning.scoring.scoring_factory import ScoringFactory
from core.model_tuning.scoring.abstract_scoring import AbstractScoring
from core.data_tuning_optimizer.config.data_tuning_config import DataTuningSetup


class DataTunerWrapperModel:
    def __init__(self, config_gui: DataTuningSetup, xy):
        self.config_gui = config_gui
        self.xy = xy
        self.scorer: AbstractScoring = ScoringFactory.scoring_factory(
            config_gui.wrapper_scoring
        )
        self.model: AbstractMLModel = ModelFactory.model_factory(
            config_gui.wrapper_model
        )

    # def prepare_train_test_feature_target(self, xy):
    #     self.x, self.y = split_target_features(self.config_gui.name_of_target, xy)
    #     self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y,
    #                                                                             test_size=0.25)
    #
    # def redo_identical_splitting(self, x_processed: pd.DataFrame):
    #     x_train_processed = x_processed.loc[self.y_train.index]
    #     x_test_processed = x_processed.loc[self.y_test.index]
    #     return x_train_processed, x_test_processed

    def train_score(self, x_train, x_test, y_train, y_test):
        self.model.fit(x_train, y_train)
        score = self.scorer.score(self.model, x_test, y_test)

        return score

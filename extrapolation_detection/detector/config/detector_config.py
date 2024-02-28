from core.util.abstract_config import BaseConfig
class DetectorConfig(BaseConfig):

    def __init__(self):
        self.detectors: list[str] = ["KNN"]
        self.detector_experiment_name: str = "test"

        # for splitting train / test / validation
        self.use_test_for_validation: bool = True # appends test data to validation data
        self.use_train_for_validation: bool = False # appends train data to validation data
        self.use_remaining_for_validation: bool = False # appends remaining data to validation data

        # for tuning
        self.tuning_bool: bool = True # if True, the detector will be tuned

        # for scoring
        self.beta_f_score: float = 1 # beta value for F-score

        # for training
        self.outlier_fraction: float = 0.05 # irrelevant for tuned detectors

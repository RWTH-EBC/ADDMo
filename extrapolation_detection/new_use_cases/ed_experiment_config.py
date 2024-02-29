from core.util.abstract_config import BaseConfig
from core.s3_model_tuning.config.model_tuning_config import ModelTuningSetup

from extrapolation_detection.detector.config.detector_config import DetectorConfig

class ExtrapolationExperimentConfig(BaseConfig):

    def __init__(self):
        # Global Variables ########################################
        self.simulation_data_name: str = "Carnot_mid"

        self.experiment_name: str = "Carnot_Test2"

        self.name_of_target: str = "$\dot{Q}_{heiz}$ in kW"

        # split data

        # Specify data indices used for training, validation and testing of regressor
        self.train_val_test: list = list(range(0, 744))
        self.val_fraction: float = 0.1
        self.test_fraction: float = 0.1
        self.shuffle: bool = True

        # True Validity Domain Variables ########################################
        self.true_outlier_fraction: float = 0.1

        # Model Tuning Variables ########################################
        # self.model_tuning_config_path: str = r"D:\\04_GitRepos\\addmo-extra\\core\\s3_model_tuning\\config\\model_tuning_config.yaml"
        # self.model_tuning_config = ModelTuningSetup().load_yaml_to_class(self.model_tuning_config_path)
        self.model_tuning_config = ModelTuningSetup()

        # Gridding Variables ########################################
        self.grid_points_per_axis = 100
        self.system_simulation = "carnot"

        # Extrapolation Detector Variables ###############################
        self.detector_config = DetectorConfig()
        # self.extrapolation_detect_config_path: str = r"D:\\04_GitRepos\\addmo-extra\\extrapolation_detection\\new_use_cases\\config_ED.yaml"
        # self.extrapolation_detect_config = ModelTuningSetup().load_yaml_to_class(self.extrapolation_detect_config_path)
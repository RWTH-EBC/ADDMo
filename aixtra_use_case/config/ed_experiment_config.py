import os.path
from typing import Tuple, Optional, Union, List

from pydantic import BaseModel, Field

from addmo.util.definitions import ed_use_case_dir
from addmo.s3_model_tuning.config.model_tuning_config import ModelTunerConfig
from aixtra.extrapolation_detection.detector.config.detector_config import DetectorConfig
from aixtra.exploration_quantification.config.explo_quant_config import ExploQuantConfig


class ExtrapolationExperimentConfig(BaseModel):
    # experiment specifications
    simulation_data_name: str = Field(
        "Carnot_mid", description="Name of the simulation system_data"
    )
    experiment_name: str = Field("Carnot_Test", description="Name of the experiment")

    name_of_target: str = Field(
        "$\dot{Q}_{heiz}$ in kW", description="Name of the target variable"
    )

    # Specify system_data indices used for training, validation and testing of regressor
    train_val_test_period: List = Field(
        [[0, 744]],
        description="If list of list, it is the start and end of several periods, like [[0, 744], [1488, 2232]]."
                    "If list, it is directly a list of indices, like [0, 1, 2, 3, 4, 744, 745].",
    )
    val_fraction: float = Field(0.1, description="Fraction of system_data for validation")
    test_fraction: float = Field(0.1, description="Fraction of system_data for testing")
    shuffle: bool = Field(True, description="Whether to shuffle the system_data")

    true_outlier_fraction: Optional[float] = Field(
        0.1, description="Fraction of true outliers."
    )
    true_outlier_threshold_error_metric: str = Field(
        "mae",
        description="Error metric used to calculate the errors of the regressor and consequently "
                    "the unit for the true outlier threshold",
    )
    true_outlier_threshold: Optional[float] = Field(
        default=None,
        description="Threshold for true outliers. None if it should be inferred through "
        "true_outlier_fraction. If not None it will overwrite true_outlier_fraction.",
    )

    # specifications for generating the grid of artificial system_data
    grid_points_per_axis: int = Field(10, description="Number of grid points per axis")
    system_simulation: Optional[str] = Field(
        "carnot", description="System simulation type, None if no simulation"
    )

    # specifications for gradients analysis
    var4gradient: str = Field(
        "control var",
        description="Variable for which the gradient is calculated. For MPC use cases this is normally the control variable.",
    )
    correct_gradient: int = Field(
        1,
        description="Value for correct gradient. 1 for positive correlation, 0 for no correlation, -1 for negative correlation.",
    )
    gradient_zero_margin: float = Field(
        1e-6,
        description="Margin for the gradient calculation for which the gradient is considered zero",
    )

    # additional configs to be imported
    config_model_tuning: ModelTunerConfig = Field(
        ModelTunerConfig(), description="Model tuning setup, set your own config."
    )
    config_detector: DetectorConfig = Field(
        DetectorConfig(), description="Detector config, set your own config."
    )

    config_explo_quant: ExploQuantConfig = Field(
        ExploQuantConfig(),
        description="Exploration quantification config, set your own config.",
    )

    @property
    def experiment_folder(self):
        return os.path.join(ed_use_case_dir(), "results", self.experiment_name)

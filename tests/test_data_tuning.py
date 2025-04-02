import unittest
import pandas as pd
from unittest.mock import  patch, MagicMock
from addmo.util.experiment_logger import ExperimentLogger,LocalLogger, WandbLogger
from addmo.s1_data_tuning_auto.config.data_tuning_auto_config import DataTuningAutoSetup
from addmo.s1_data_tuning_auto.data_tuner_auto import DataTunerAuto
from addmo.s2_data_tuning.config.data_tuning_config import DataTuningFixedConfig
from addmo.s2_data_tuning.data_tuner_fixed import DataTunerByConfig
from addmo.util.load_save import load_data
from addmo.util.data_handling import split_target_features


class TestDataTunerAuto(unittest.TestCase):
    """
    Unit tests for data tuning using pre-defined config.
    """

    def setUp(self):
        """Set up config"""
        self.config = DataTuningAutoSetup()

    @patch("addmo.util.experiment_logger.ExperimentLogger.log_artifact") # skips actual logging behaviour and checks if its executed without errors
    def test_data_tuner(self,mock_log_artifact):
        """Test expected outputs of data tuner"""
        # Create system_data tuner
        tuner = DataTunerAuto(config=self.config)

        # Tune system_data
        tuned_x = tuner.tune_auto()
        y = tuner.y
        tuned_xy = pd.concat([y, tuned_x], axis=1, join="inner").bfill()

        # Validation
        self.assertFalse(tuned_x.empty, "Tuned data is empty")
        self.assertIsInstance(tuned_x, pd.DataFrame, "Tuned data is not a dataframe")
        self.assertFalse(y.empty, "Tuned target column is empty")
        self.assertIsInstance(y, pd.Series, "Tuned target column is not a series")
        self.assertFalse(tuned_xy.empty, "Tuned xy data is empty")
        self.assertIsInstance(tuned_xy, pd.DataFrame, "Tuned xy data is not a dataframe")

        try:
            ExperimentLogger.log_artifact(tuned_xy, name='tuned_xy', art_type='csv')

        except Exception as e:
            self.fail(f"ExperimentLogger.log_artifact raised an exception: {e}")


class TestDataTunerFixed(unittest.TestCase):
    """
    Unit tests for data tuning using pre-defined config.
    """

    def setUp(self):
        """Set up config"""
        self.config = DataTuningFixedConfig()

    @patch(
        "addmo.util.experiment_logger.ExperimentLogger.log")
    @patch(
        "addmo.util.experiment_logger.ExperimentLogger.log_artifact")  # skips actual logging behaviour and checks if its executed without errors
    def test_data_tuner(self, mock_log_artifact, mock_log):
        """Test expected outputs of data tuner"""

        # Create system_data tuner
        tuner = DataTunerByConfig(config=self.config)

        # Load system_data
        xy_raw = load_data(self.config.abs_path_to_data)
        self.assertIsInstance(xy_raw, pd.DataFrame)
        self.assertFalse(xy_raw.empty)

        x, y = split_target_features(self.config.name_of_target, xy_raw)

        # Tune the system_data
        tuned_x = tuner.tune_fixed(xy_raw)
        # Merge target and features
        xy_tuned = tuned_x.join(y)

        # Validation
        self.assertFalse(tuned_x.empty, "Tuned data is empty")
        self.assertIsInstance(tuned_x, pd.DataFrame, "Tuned data is not a dataframe")
        self.assertFalse(xy_tuned.empty, "Tuned xy data is empty")
        self.assertIsInstance(xy_tuned, pd.DataFrame, "Tuned xy data is not a dataframe")

        try:
            ExperimentLogger.log({"xy_tuned": xy_tuned.iloc[[0, 1, 2, -3, -2, -1]]})
        except Exception as e:
            self.fail(f"ExperimentLogger.log raised an exception: {e}")


        try:
            ExperimentLogger.log_artifact(xy_tuned, name='xy_tuned', art_type='csv')

        except Exception as e:
            self.fail(f"ExperimentLogger.log_artifact raised an exception: {e}")


if __name__ == "__main__":
    unittest.main()
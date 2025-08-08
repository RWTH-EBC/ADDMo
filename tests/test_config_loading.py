import unittest
from pydantic import ValidationError
from addmo.s1_data_tuning_auto.config.data_tuning_auto_config import DataTuningAutoSetup
from addmo.s2_data_tuning.config.data_tuning_config import  DataTuningFixedConfig
from addmo.s3_model_tuning.config.model_tuning_config import  ModelTunerConfig, ModelTuningExperimentConfig

class TestConfigLoading(unittest.TestCase):
    """
    Use this test if new pydantic config classes are added to ADDMo
    """
    def test_required_fields_data_config(self):
        """
        Check if fields necessary to run the training and plotting utilities for data tuning are added to config classes.
        """
        required_fields = ["abs_path_to_data", "name_of_target","name_of_raw_data","name_of_tuning"]
        config_classes = [DataTuningAutoSetup, DataTuningFixedConfig]  # Add new config classes here
        for ConfigClass in config_classes:
            with self.subTest(config=ConfigClass.__name__):
                config = ConfigClass()
                for field in required_fields:
                    self.assertTrue(
                        hasattr(config, field),
                        msg=f"{ConfigClass.__name__} is missing required field '{field}'"
                    )
    def test_required_fields_model_config(self):
        """
        Check if fields necessary to run the model tuner are added to model experiment config class.
        """
        required_fields=["abs_path_to_data", "name_of_data_tuning_experiment","name_of_model_tuning_experiment","name_of_target"]
        config_classes = [ModelTuningExperimentConfig]  # Add model tuning experiment classes here
        for ConfigClass in config_classes:
            with self.subTest(config=ConfigClass.__name__):
                config = ConfigClass()
                for field in required_fields:
                    self.assertTrue(
                        hasattr(config, field),
                        msg=f"{ConfigClass.__name__} is missing required field '{field}'"
                    )

    def test_required_fields_model_tuner(self):
        """
        Check if fields necessary to run the model tuner are added to config class.
        """
        required_fields = ["models","trainings_per_model"]
        config_classes = [ModelTunerConfig]
        for ConfigClass in config_classes:
            with self.subTest(config=ConfigClass.__name__):
                config = ConfigClass()
                for field in required_fields:
                    self.assertTrue(
                        hasattr(config, field),
                        msg=f"{ConfigClass.__name__} is missing required field '{field}'"
                    )

    def test_data_tuning_config_enforces_str_types(self):
        """
        For every pydantic config that should use str fields, injecting a non-str
        must raise ValidationError.
        """
        config_classes = [DataTuningAutoSetup, DataTuningFixedConfig]
        for ConfigClass in config_classes:
            with self.subTest(config=ConfigClass.__name__):
                valid_kwargs = {
                    name: "dummy"
                    for name, field in ConfigClass.model_fields.items()
                    if field.annotation == str
                }
                # Test each str field rejects non-str
                for name, field in ConfigClass.model_fields.items():
                    if field.annotation == str:
                        bad_kwargs = valid_kwargs.copy()
                        bad_kwargs[name] = 123
                        with self.assertRaises(ValidationError, msg=f"{ConfigClass.__name__}.{name} did not validate"):
                            ConfigClass(**bad_kwargs)
if __name__ == "__main__":
    unittest.main()

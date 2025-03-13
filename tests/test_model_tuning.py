import unittest
import pandas as pd
from addmo.s3_model_tuning.models.model_factory import ModelFactory
from addmo.s3_model_tuning.config.model_tuning_config import ModelTuningExperimentConfig
from addmo.s3_model_tuning.config.model_tuning_config import ModelTunerConfig
from addmo.s3_model_tuning.model_tuner import ModelTuner
from addmo.util.load_save import load_data
from addmo.util.load_save import load_config_from_json
from addmo.util.data_handling import split_target_features


class TestModelTuner(unittest.TestCase):
    """
    Unit tests for ModelTuner using pre-defined config.
    """
    def setUp(self):
        self.config = ModelTuningExperimentConfig()

    def test_model_tuning(self):
        model_tuner = ModelTuner(config=self.config.config_model_tuner)

        # Load the system_data
        xy_tuned = load_data(self.config.abs_path_to_data)

        # Select training and validation period
        xy_tuned_train_val = xy_tuned.loc[self.config.start_train_val:self.config.stop_train_val]
        x_train_val, y_train_val = split_target_features(self.config.name_of_target, xy_tuned_train_val)

        # Tune the models
        model_dict = model_tuner.tune_all_models(x_train_val, y_train_val)

        # Get the best model
        best_model_name = model_tuner.get_best_model_name(model_dict)
        best_model = model_tuner.get_model(model_dict, best_model_name)

        # Validation
        self.assertFalse(xy_tuned.empty, "loaded data is empty")
        self.assertIsInstance(xy_tuned, pd.DataFrame, "load_data() not returning a pd.DataFrame")
        self.assertIsInstance(xy_tuned_train_val, pd.DataFrame,"train_val data is no longer a pd.DataFrame")
        self.assertFalse(xy_tuned_train_val.empty, "train_val data is empty")
        self.assertIsInstance(model_dict, dict, "tune_all_models() should return a dict")
        self.assertSetEqual(set(model_dict.keys()), set(model_tuner.config.models), "Returned models do not match expected models")

        for model_name, model in model_dict.items():
            self.assertIsNotNone(model, f"Model {model_name} cannot be None")
            expected_model_class = ModelFactory.model_factory(model_name).__class__
            self.assertIsInstance(model, expected_model_class, f"Model should be {expected_model_class} but got {type(model)}")

        self.assertIsNotNone(best_model, "best_model is None")

if __name__ == "__main__":
    unittest.main()
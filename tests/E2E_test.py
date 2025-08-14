import unittest
import os
import shutil
import pandas as pd
from pathlib import Path
from addmo.util.load_save import load_config_from_json, load_data
from addmo.util.definitions import results_dir_data_tuning, results_dir_model_tuning, return_best_model
from addmo_examples.executables.exe_data_tuning_fixed import exe_data_tuning_fixed
from addmo_examples.executables.exe_model_tuning import exe_model_tuning
from addmo_examples.executables.exe_data_insights import exe_carpet_plots
from addmo.s2_data_tuning.config.data_tuning_config import DataTuningFixedConfig
from addmo.s3_model_tuning.config.model_tuning_config import ModelTuningExperimentConfig, ModelTunerConfig
from addmo.util.load_save_utils import root_dir

# This unittest is to test whether all the executables work compatibly with each other. In case new configs are added, the user would have to create executables which tests
# the created tuning based on configs, then the model tuning and insights are handled here. Similarly, newly added models can also be tested for existing tuned data and insights.


# This is manual end to end test, please ensure to define the paths to new configs in order to test full pipeline

class TestAddmoEndToEnd(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Test the functionality of addmo_examples in a complete pipeline
        """
        # Define data tuning and model tuning setup
        cls.data_config_path = os.path.join(root_dir(),
            "addmo", "s2_data_tuning", "config", "data_tuning_config.json"
        )
        cls.data_config = load_config_from_json(cls.data_config_path, DataTuningFixedConfig)
        if hasattr(cls.data_config, "abs_path_to_data"):
            filename = Path(cls.data_config.abs_path_to_data).name
            cls.data_config.abs_path_to_data = str(
                Path(root_dir()) / "addmo_examples" / "raw_input_data" / filename
            )
        cls.model_tuner_config_path = os.path.join(root_dir(), "addmo", "s3_model_tuning", "config", "model_tuner_config.json")
        cls.model_exp_config_path = os.path.join(root_dir(), "addmo", "s3_model_tuning", "config", "model_tuner_experiment_config.json")
        cls.model_config = load_config_from_json(cls.model_exp_config_path, ModelTuningExperimentConfig)
        cls.model_tuner_config = load_config_from_json(cls.model_tuner_config_path, ModelTunerConfig)

    def test_full_pipeline(self):

        # Test Data Tuning executable file
        exe_data_tuning_fixed()
        data_dir = results_dir_data_tuning(self.data_config)
        tuned_csv = os.path.join(data_dir, "tuned_xy_fixed.csv")
        self.assertTrue(os.path.exists(tuned_csv), "Tuned data file missing.")
        xy_tuned = load_data(tuned_csv)
        self.assertFalse(xy_tuned.empty, "Tuned xy data is empty")
        self.assertIsInstance(xy_tuned, pd.DataFrame, "Tuned xy data is not a dataframe")

        # Use the tuned data as input data
        self.model_config.abs_path_to_data = tuned_csv

        # Test Model Tuning executable file
        exe_model_tuning(config_exp= self.model_config, config_tuner= self.model_tuner_config)
        model_dir = results_dir_model_tuning(self.model_config)
        best_model_path = return_best_model(model_dir)
        self.assertTrue(os.path.exists(best_model_path), "Best model not saved.")

        # Use the results to test a plotting functionality
        exe_carpet_plots(
            dir=model_dir,
            plot_name="carpet_plot",
            plot_dir=model_dir,
            path_to_regressor=best_model_path,
            save=True
        )
        carpet_plot_path = os.path.join(model_dir, "carpet_plot.pdf")
        self.assertTrue(os.path.exists(carpet_plot_path), "Carpet plot not generated.")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(results_dir_data_tuning(cls.data_config), ignore_errors=True)
        shutil.rmtree(results_dir_model_tuning(cls.model_config), ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

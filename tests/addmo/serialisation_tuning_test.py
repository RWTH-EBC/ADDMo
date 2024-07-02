import unittest
import pandas as pd
import os
from addmo.s3_model_tuning.models.scikit_learn_models import ScikitMLP, ScikitLinearReg
from addmo.s3_model_tuning.models.model_factory import ModelFactory
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from addmo.s3_model_tuning.models.keras_models import SciKerasSequential
from addmo.util.definitions import root_dir
from addmo.s3_model_tuning.config.model_tuning_config import ModelTunerConfig
from addmo.s3_model_tuning.model_tuner import ModelTuner

def test_save_load_model(model, dir_path, X_test, y_test, file_type='joblib'):
    """
    Test saving and loading of model.

    Args:
    - model: The model instance to be tested.
    - dir_path: The directory path to save and load the model files.
    - X_test: Test features.
    - y_test: Test labels.
    - file_type: Type of file to save the model. Default is 'joblib'.
    """
    # Save the model
    model.save_regressor(dir_path, file_type=file_type)

    # Load the model
    loaded_model = ModelFactory().load_model(os.path.join(dir_path, f"{type(model).__name__}.{file_type}"))

    # Make predictions
    y_pred_loaded = loaded_model.predict(X_test)

    # Calculate R-squared
    r_squared_loaded = r2_score(y_test, y_pred_loaded)

    # Check if R-squared is a number
    assert isinstance(r_squared_loaded, (int, float))

    # Clean up saved files
    os.remove(os.path.join(dir_path, f"{type(model).__name__}.{file_type}"))
    os.remove(os.path.join(dir_path, f"{type(model).__name__}{'_metadata.json'}"))

    return loaded_model


class TestModels(unittest.TestCase):

    def setUp(self):
        # Load and prepare system_data
        data = fetch_california_housing()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['price'] = pd.Series(data.target)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(df.iloc[:, :-1], df['price'], test_size=0.2, random_state=42)
        self.dir_path = os.path.join(root_dir(), "0000_testfiles")

    def test_linear_regression(self):
        # Testing Linear Regression model saved in .joblib format
        model = ScikitLinearReg()
        model.fit(self.X_train, self.y_train)

        # Testing saving and loading of model
        loaded_model = test_save_load_model(model, self.dir_path, self.X_test, self.y_test)
        print('loaded model is : ', loaded_model)


    def test_scikit_mlp(self):
        # Testing MLP model saved in .onnx format
        model = ScikitMLP()
        model.fit(self.X_train, self.y_train)

        # Testing saving and loading of model
        loaded_model = test_save_load_model(model, self.dir_path, self.X_test, self.y_test,  file_type='onnx')
        print('loaded model is : ', loaded_model)


    def test_keras_model(self):
        # Testing Keras model saved in .keras format
        model = SciKerasSequential()
        model.fit(self.X_train, self.y_train)

        # Testing saving and loading of model
        loaded_model = test_save_load_model(model, self.dir_path, self.X_test, self.y_test, file_type='h5')
        print('loaded model is : ', loaded_model)

class TestKerasOptunaTuning(unittest.TestCase):

    def setUp(self):
        # Load and prepare system_data
        data = fetch_california_housing()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['price'] = pd.Series(data.target)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(df.iloc[:, :-1], df['price'], test_size=0.2, random_state=42)
        self.dir_path = os.path.join(root_dir(), "0000_testfiles")
        self.config = self.setup_model_tuning_config()

    def setup_model_tuning_config(self):
        # Configures and returns a ModelTunerConfig instance
        return ModelTunerConfig(
            models=["SciKerasSequential"],
            hyperparameter_tuning_type="OptunaTuner",
            hyperparameter_tuning_kwargs={"n_trials": 2},
            validation_score_mechanism="cv",
            validation_score_splitting="KFold",
            validation_score_metric="neg_mean_squared_error"
        )

    def test_model_tuning(self):
        # Create a tuner instance and perform tuning
        tuner = ModelTuner(self.config)
        model_dict = tuner.tune_all_models(x_train_val=self.X_train, y_train_val=self.y_train)
        best_model_name = tuner.get_best_model_name(model_dict)
        best_model = tuner.get_model(model_dict, best_model_name)
        score = tuner.get_model_validation_score(model_dict, best_model_name)

        print('Best Model is: ', best_model_name, best_model)
        print('Score is: ', score)
        self.assertIsInstance(score, float)


if __name__ == '__main__':
    unittest.main()

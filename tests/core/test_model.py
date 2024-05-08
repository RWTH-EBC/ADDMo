import unittest
import pandas as pd
import numpy as np
import os
from core.s3_model_tuning.models.scikit_learn_models import BaseScikitLearnModel
from core.s3_model_tuning.models.scikit_learn_models import ScikitMLP, LinearReg
from core.s3_model_tuning.models.model_factory import ModelFactory
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from core.s3_model_tuning.models.abstract_model import AbstractMLModel
from core.s3_model_tuning.models.keras_model import BaseKerasModel
from core.util.definitions import root_dir


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


class TestModels(unittest.TestCase):

    def setUp(self):
        # Load and prepare data
        data = fetch_california_housing()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['price'] = pd.Series(data.target)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(df.iloc[:, :-1], df['price'],
                                                                                test_size=0.2, random_state=42)
        self.dir_path = os.path.join(root_dir(), "0000_testfiles")



    def test_linear_regression(self):
        # Testing Linear Regression model saved in .joblib format
        model = LinearReg()
        model.fit(self.X_train, self.y_train)

        # Testing saving and loading of model
        test_save_load_model(model, self.dir_path, self.X_test, self.y_test)

    def test_scikit_mlp(self):
        # Testing MLP model saved in .onnx format
        model = ScikitMLP()
        model.fit(self.X_train, self.y_train)

        # Testing saving and loading of model
        test_save_load_model(model, self.dir_path, self.X_test, self.y_test,  file_type='onnx')


    def test_keras_model(self):
        # Testing Keras model saved in .keras format
        model = BaseKerasModel()
        model.fit(self.X_train, self.y_train)

        # Testing saving and loading of model
        test_save_load_model(model, self.dir_path, self.X_test, self.y_test, file_type='keras')


if __name__ == '__main__':
    unittest.main()
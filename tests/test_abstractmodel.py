import unittest
import tempfile
import os
import pandas as pd
import numpy as np
from addmo.s3_model_tuning.models.abstract_model import AbstractMLModel
from addmo.s3_model_tuning.models.keras_models import BaseKerasModel
from addmo.s3_model_tuning.models.scikit_learn_models import BaseScikitLearnModel
from addmo.s3_model_tuning.models.abstract_model import ModelMetadata
from addmo.s3_model_tuning.models.model_factory import ModelFactory


def get_subclasses(base_class):
    """
    Dynamically get all the subclasses which contain the models for the given base class.
    """
    subclasses = set()
    work = [base_class]
    while work:
        parent = work.pop()
        for child in parent.__subclasses__():
            if child not in subclasses:
                subclasses.add(child)
                work.append(child)
    return list(subclasses)

def train_and_check_model(self, model, x_sample, y_sample):
    """
    Test the fit and predict functionality of the model.
    """
    model.fit(x_sample, y_sample)
    predictions = model.predict(x_sample)
    self.assertEqual(len(predictions), len(y_sample))
    self.assertIsInstance(predictions, np.ndarray)

class TestBaseMLModel(unittest.TestCase):
    """
    Unit tests for base class models.
    """

    base_class = BaseScikitLearnModel  # Change this to test different base classes

    @classmethod
    def setUpClass(cls):
        """
        Find all subclasses of the base class.
        """

        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.subclasses = get_subclasses(cls.base_class)

        if not cls.subclasses:
            raise ValueError(f"No subclasses found for {cls.base_class.__name__}")

    @classmethod
    def tearDownClass(cls):
        """
        Cleanup temp directory after tests.
        """
        cls.temp_dir.cleanup()


    def test_all_models(self):
        """
        Test all registered models that are subclasses of AbstractMLModel.
        """

        # Ensure regressor is not None
        for model_class in self.subclasses:
            with self.subTest(model=model_class.__name__):

                model = model_class()  # Instantiate model
                print(f"\n Testing model {model_class.__name__}")

                self.assertIsNotNone(model.regressor, f"{model_class.__name__} should have a regressor")

                x_sample = pd.DataFrame(np.random.rand(100, 2), columns=["A", "B"])
                y_sample = pd.Series(np.random.rand(100), name = "Target")

                train_and_check_model(self, model, x_sample, y_sample)

                # Test model serialization
                file_type= model.save_regressor(self.temp_dir.name, "test_model")

                path_to_regressor= os.path.join(self.temp_dir.name, f'test_model.{file_type}')
                test_regressor: AbstractMLModel = ModelFactory.load_model(path_to_regressor)
                # Test metadata
                self.assertIsInstance(model.metadata, ModelMetadata, "Meta data not defined properly")
                self.assertEqual(model.metadata.addmo_class, model_class.__name__, "Model class mismatch")
                self.assertIsInstance(test_regressor, type(model), "Loaded model class mismatch")


if __name__ == "__main__":
    unittest.main()


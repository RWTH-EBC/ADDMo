import unittest
import tempfile
import os
import pandas as pd
import numpy as np
from addmo.s3_model_tuning.models.abstract_model import AbstractMLModel
from addmo.s3_model_tuning.models.keras_models import BaseKerasModel
from addmo.s3_model_tuning.models.scikit_learn_models import BaseScikitLearnModel
from addmo.s3_model_tuning.models.abstract_model import ModelMetadata
from aixtra.util import loading_saving_aixtra



def get_subclasses(base_class):
    """Dynamically get all the subclasses which contain the models for the given base class"""
    return base_class.__subclasses__()

class TestBaseMLModel(unittest.TestCase):
    """Unit tests for Base class models"""

    base_class = BaseKerasModel  # Change this to test different base classes

    @classmethod
    def setUpClass(cls):
        """Find all subclasses of the base class."""

        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.subclasses = get_subclasses(cls.base_class)
        if not cls.subclasses:
            raise ValueError(f"No subclasses found for {cls.base_class.__name__}")

    @classmethod
    def tearDownClass(cls):
        """Cleanup temp directory after tests."""
        cls.temp_dir.cleanup()

    def test_all_models(self):
        """Test all registered models that are subclasses of AbstractMLModel."""

        # Ensure regressor is not None
        for model_class in self.subclasses:
            with self.subTest(model=model_class.__name__):
                model = model_class()  # Instantiate model
                self.assertIsNotNone(model.regressor, f"{model_class.__name__} should have a regressor")

        x_sample = pd.DataFrame(np.random.rand(10, 2), columns=["A", "B"])
        y_sample = pd.Series(np.random.rand(10), name = "Target")


        model.fit(x_sample, y_sample)
        predictions = model.predict(x_sample)

        # Test training and prediction
        self.assertEqual(len(predictions), len(y_sample))
        self.assertIsInstance(predictions, np.ndarray)

        # Test model serialization

        model.save_regressor(self.temp_dir.name, "test_model")
        test_regressor: AbstractMLModel = loading_saving_aixtra.load_regressor("test_model", directory=os.path.join(self.temp_dir.name))
        # Test metadata
        self.assertIsInstance(model.metadata, ModelMetadata, "Meta data not defined properly")
        self.assertEqual(model.metadata.addmo_class, model_class.__name__, "Model class mismatch")
        self.assertIsInstance(test_regressor, type(model), "Loaded model class mismatch")



if __name__ == "__main__":
    unittest.main()


import unittest
import os
import keras
import pandas as pd
from tensorflow.keras.models import Sequential
from unittest.mock import patch, MagicMock
from core.s3_model_tuning.models.keras_model import BaseKerasModel
from tensorflow.keras.optimizers import SGD


class TestBaseKerasModel(unittest.TestCase):
    def setUp(self):
        self.input_shape = 4
        self.output_shape = 3
        self.model = BaseKerasModel(self.input_shape, self.output_shape)

    def test_init(self):
        self.assertEqual(self.model.input_shape, self.input_shape)
        self.assertEqual(self.model.output_shape, self.output_shape)
        self.assertIsInstance(self.model.regressor, Sequential)

    def test_compile_model(self):
        self.model.compile_model()
        self.assertTrue(hasattr(self.model, 'learning_rate'))
        expected_optimizer = SGD()  # Instantiate SGD optimizer
        self.assertIsInstance(self.model.regressor.optimizer, type(expected_optimizer))
        self.assertEqual(self.model.regressor.loss, 'categorical_crossentropy')
        #self.assertEqual(keras.metrics.Accuracy(), self.model.regressor.metrics)

    def test_fit(self):
        # Mock input data as DataFrame
        x_train_data = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]  # Example data
        x_train = pd.DataFrame(x_train_data)

        y_train_data = [[1, 0, 0], [0, 1, 0]]  # Example one-hot encoded labels
        y_train = pd.DataFrame(y_train_data)
        self.model.compile_model()
        # Call the fit method
        history = self.model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=None)
        print(history)

        # Assert that the history object returned is not None
        self.assertIsNotNone(history)


    @patch('core.s3_model_tuning.models.keras_model.json.dump')
    @patch('core.s3_model_tuning.models.keras_model.os.path')
    @patch('core.s3_model_tuning.models.keras_model.get_commit_id', return_value='123456')
    def test_save_regressor(self, mock_get_commit_id, mock_os_path, mock_json_dump):
        mock_os_path.join.return_value = 'mock_path'
        self.model._save_metadata = MagicMock()
        self.model.save_regressor('directory', 'filename')
        self.model._save_metadata.assert_called_once_with('directory', 'filename')
        self.model.regressor.save.assert_called_once_with('mock_path', overwrite=True)
        mock_json_dump.assert_called_once()

    @patch('core.s3_model_tuning.models.keras_model.keras.models.load_model')
    @patch('core.s3_model_tuning.models.keras_model.load_metadata')
    def test_load_regressor(self, mock_load_metadata, mock_load_model):
        mock_load_metadata.return_value = {'input_shape': (4,), 'output_shape': 3}
        mock_load_model.return_value = MagicMock()
        self.model.load_regressor('regressor')
        self.assertIsInstance(self.model.regressor, MagicMock)
        mock_load_model.assert_called_once_with('regressor')

    def test_save_metadata(self):
        # Mock data

        x_train_data = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]  # Example data
        self.model.x = pd.DataFrame(x_train_data)

        y_train_data = [[1, 0, 0], [0, 1, 0]]  # Example one-hot encoded labels
        self.model.y = pd.DataFrame(y_train_data)
        self.model._save_metadata('directory', 'filename')
        self.assertEqual(self.model.metadata.addmo_class, 'BaseKerasModel')
        self.assertEqual(self.model.metadata.addmo_commit_id, '123456')
        self.assertEqual(self.model.metadata.library, 'keras')
        self.assertEqual(self.model.metadata.library_model_type, 'Sequential')
        self.assertEqual(self.model.metadata.input_shape, self.input_shape)
        self.assertEqual(self.model.metadata.output_shape, self.output_shape)
        self.assertEqual(self.model.metadata.target_name, list(self.model.y.columns))
        self.assertEqual(self.model.metadata.features_ordered, list(self.model.x.columns))


if __name__ == '__main__':
    unittest.main()
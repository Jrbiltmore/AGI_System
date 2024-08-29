
import unittest
import numpy as np
from core.main_model import AGIModel
from core.preprocessing import DataPreprocessor
from core.postprocessing import PostProcessor
from data.data_loader import DataLoader
from data.augmentation_tools import DataAugmentation
from training.loss_functions import CustomLossFunctions
import tensorflow as tf

class TestAGIModel(unittest.TestCase):
    def setUp(self):
        self.model = AGIModel(input_shape=(10,), output_shape=2)

    def test_model_structure(self):
        self.assertEqual(len(self.model.model.layers), 6, "Model should have 6 layers")
    
    def test_model_training(self):
        X_train = np.random.rand(100, 10)
        y_train = np.random.randint(0, 2, 100)
        history = self.model.train(X_train, y_train, X_train, y_train, epochs=1)
        self.assertIn('loss', history.history, "Training history should contain 'loss'")

class TestDataPreprocessor(unittest.TestCase):
    def setUp(self):
        self.preprocessor = DataPreprocessor()

    def test_fit_transform(self):
        X = np.array([[1, 2, np.nan], [4, np.nan, 6], [7, 8, 9]])
        X_transformed = self.preprocessor.fit_transform(X)
        self.assertFalse(np.isnan(X_transformed).any(), "There should be no NaN values after fit_transform")

class TestPostProcessor(unittest.TestCase):
    def setUp(self):
        self.post_processor = PostProcessor()

    def test_compute_metrics(self):
        y_true = [1, 0, 1, 1, 0, 1, 0, 0]
        y_pred = [1, 0, 1, 0, 0, 1, 0, 1]
        metrics = self.post_processor.compute_metrics(y_true, y_pred)
        self.assertIn('accuracy', metrics, "Metrics should contain 'accuracy'")

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.data_loader = DataLoader(data_path='/mnt/data/AGI_System/data/raw/')

    def test_load_csv(self):
        try:
            df = self.data_loader.load_csv('sample_data.csv')
            self.assertIsNotNone(df, "Dataframe should not be None after loading CSV")
        except FileNotFoundError:
            self.skipTest("CSV file not found, skipping test.")

class TestDataAugmentation(unittest.TestCase):
    def setUp(self):
        self.augmentor = DataAugmentation()

    def test_oversample_minority_class(self):
        X = np.random.rand(100, 10)
        y = np.array([0] * 90 + [1] * 10)
        X_upsampled, y_upsampled = self.augmentor.oversample_minority_class(X, y, minority_class=1)
        self.assertEqual(sum(y_upsampled == 1), sum(y_upsampled == 0), "Minority class should be oversampled to match the majority class")

class TestCustomLossFunctions(unittest.TestCase):
    def test_focal_loss(self):
        y_true = tf.constant([[1, 0, 0], [0, 1, 0]], dtype=tf.float32)
        y_pred = tf.constant([[0.9, 0.1, 0], [0.1, 0.8, 0.1]], dtype=tf.float32)
        focal_loss = CustomLossFunctions.focal_loss()
        loss = focal_loss(y_true, y_pred)
        self.assertIsInstance(loss, tf.Tensor, "Focal loss should return a tensor")

if __name__ == '__main__':
    unittest.main()

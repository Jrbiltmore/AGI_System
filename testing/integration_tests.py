
import unittest
import numpy as np
from core.main_model import AGIModel
from core.preprocessing import DataPreprocessor
from core.postprocessing import PostProcessor
from data.data_loader import DataLoader
from data.augmentation_tools import DataAugmentation
from training.train import train_model, load_and_preprocess_data, augment_data
import tensorflow as tf

class TestModelTrainingIntegration(unittest.TestCase):
    def setUp(self):
        self.data_loader = DataLoader(data_path='/mnt/data/AGI_System/data/raw/')
        self.model_path = '/mnt/data/AGI_System/training/checkpoints/best_model.h5'
        self.X, self.y = load_and_preprocess_data()
        self.X_augmented, self.y_augmented = augment_data(self.X, self.y)

    def test_end_to_end_training(self):
        X_train, X_val, y_train, y_val = DataPreprocessor.split_data(pd.DataFrame(self.X_augmented), 'target', test_size=0.2)
        history, model = train_model(X_train, y_train, X_val, y_val)
        self.assertTrue(len(history.history['loss']) > 0, "Training should produce a loss history")
        self.assertTrue(os.path.exists(self.model_path), "Trained model should be saved at specified path")

    def test_model_inference(self):
        model = AGIModel(input_shape=(self.X_augmented.shape[1],), output_shape=len(np.unique(self.y_augmented)))
        model.load_model(self.model_path)
        y_pred = np.argmax(model.predict(self.X), axis=1)
        self.assertEqual(y_pred.shape, self.y.shape, "Predicted outputs should match the shape of true labels")

class TestDataPipelineIntegration(unittest.TestCase):
    def setUp(self):
        self.data_loader = DataLoader(data_path='/mnt/data/AGI_System/data/raw/')
        self.preprocessor = DataPreprocessor()
        self.augmentor = DataAugmentation()

    def test_data_loading_and_preprocessing(self):
        try:
            df = self.data_loader.load_csv('sample_data.csv')
            X = df.drop('target', axis=1)
            y = df['target']
            X_processed = self.preprocessor.fit_transform(X)
            self.assertFalse(np.isnan(X_processed).any(), "Preprocessed data should not contain NaN values")
        except FileNotFoundError:
            self.skipTest("CSV file not found, skipping test.")

    def test_data_augmentation(self):
        X = np.random.rand(100, 10)
        y = np.array([0] * 90 + [1] * 10)
        X_upsampled, y_upsampled = self.augmentor.oversample_minority_class(X, y, minority_class=1)
        self.assertEqual(sum(y_upsampled == 1), sum(y_upsampled == 0), "Data augmentation should balance class distribution")

class TestEndToEndModelEvaluation(unittest.TestCase):
    def setUp(self):
        self.post_processor = PostProcessor()
        self.model_path = '/mnt/data/AGI_System/training/checkpoints/best_model.h5'
        self.X_test, self.y_test = load_and_preprocess_data()

    def test_model_evaluation_and_postprocessing(self):
        model = AGIModel(input_shape=(self.X_test.shape[1],), output_shape=len(np.unique(self.y_test)))
        model.load_model(self.model_path)
        y_pred = np.argmax(model.predict(self.X_test), axis=1)
        metrics = self.post_processor.compute_metrics(self.y_test, y_pred)
        self.assertIn('accuracy', metrics, "Post-processed results should contain accuracy metric")

if __name__ == '__main__':
    unittest.main()

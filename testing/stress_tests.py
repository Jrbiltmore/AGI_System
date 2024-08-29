
import unittest
import numpy as np
from core.main_model import AGIModel
import tensorflow as tf
import logging
from time import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TestModelStress(unittest.TestCase):
    def setUp(self):
        self.model = AGIModel(input_shape=(1000,), output_shape=10)
        self.large_input = np.random.rand(100000, 1000)  # Large input to test stress handling
        self.large_output = np.random.randint(0, 10, 100000)

    def test_large_batch_training(self):
        start_time = time()
        self.model.train(self.large_input, self.large_output, self.large_input, self.large_output, batch_size=512, epochs=1)
        end_time = time()
        logging.info(f"Training time for large batch size: {end_time - start_time} seconds")
        self.assertTrue((end_time - start_time) < 600, "Training on large data should complete within reasonable time")

    def test_memory_usage_during_inference(self):
        try:
            start_time = time()
            predictions = self.model.predict(self.large_input)
            end_time = time()
            logging.info(f"Inference time for large input: {end_time - start_time} seconds")
            self.assertTrue(predictions.shape[0] == self.large_input.shape[0], "Predictions should match the input batch size")
        except MemoryError:
            self.fail("MemoryError: Model failed to handle large input for inference.")

    def test_model_scalability(self):
        input_sizes = [1000, 10000, 50000, 100000]
        for size in input_sizes:
            X = np.random.rand(size, 1000)
            y = np.random.randint(0, 10, size)
            try:
                self.model.train(X, y, X, y, batch_size=128, epochs=1)
                logging.info(f"Model trained successfully on input size: {size}")
            except Exception as e:
                self.fail(f"Model failed to scale with input size {size}: {str(e)}")

if __name__ == '__main__':
    unittest.main()

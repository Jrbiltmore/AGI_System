
import numpy as np
import tensorflow as tf
from core.main_model import AGIModel
from core.postprocessing import PostProcessor
from data.data_loader import DataLoader
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data():
    data_loader = DataLoader(data_path='/mnt/data/AGI_System/data/raw/')
    df_test = data_loader.load_csv('test_data.csv')
    X_test = df_test.drop('target', axis=1).values
    y_test = df_test['target'].values
    logging.info("Test data loaded successfully")
    return X_test, y_test

def load_trained_model(model_path):
    model = AGIModel(input_shape=(X_test.shape[1],), output_shape=len(np.unique(y_test)))
    model.load_model(model_path)
    logging.info(f"Model loaded from {model_path}")
    return model

def evaluate_model(model, X_test, y_test):
    results = model.evaluate(X_test, y_test)
    logging.info(f"Model evaluation completed with results: {results}")
    return results

def postprocess_results(y_true, y_pred):
    post_processor = PostProcessor()
    metrics = post_processor.compute_metrics(y_true, y_pred)
    post_processor.plot_confusion_matrix(metrics['confusion_matrix'], class_names=np.unique(y_true))
    post_processor.save_results(metrics, '/mnt/data/AGI_System/testing/results/evaluation_metrics.txt')
    logging.info("Post-processing and result saving completed")
    return metrics

def main():
    X_test, y_test = load_data()
    model = load_trained_model('/mnt/data/AGI_System/training/checkpoints/best_model.h5')
    y_pred = np.argmax(model.predict(X_test), axis=1)
    metrics = postprocess_results(y_test, y_pred)
    logging.info(f"Final evaluation metrics: {metrics}")

if __name__ == "__main__":
    main()

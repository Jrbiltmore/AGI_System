
import os
import json
import numpy as np
import pandas as pd
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FileUtils:
    @staticmethod
    def create_directory(directory_path):
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            logging.info(f"Directory created at: {directory_path}")
        else:
            logging.info(f"Directory already exists: {directory_path}")

    @staticmethod
    def save_json(data, file_path):
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
        logging.info(f"Data saved to JSON file: {file_path}")

    @staticmethod
    def load_json(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        logging.info(f"Data loaded from JSON file: {file_path}")
        return data

    @staticmethod
    def save_numpy_array(array, file_path):
        np.save(file_path, array)
        logging.info(f"Numpy array saved to file: {file_path}")

    @staticmethod
    def load_numpy_array(file_path):
        array = np.load(file_path)
        logging.info(f"Numpy array loaded from file: {file_path}")
        return array

class DataUtils:
    @staticmethod
    def split_data(df, target_column, test_size=0.2, random_state=42):
        from sklearn.model_selection import train_test_split
        X = df.drop(columns=[target_column])
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        logging.info(f"Data split into training and testing sets with test size = {test_size}")
        return X_train, X_test, y_train, y_test

    @staticmethod
    def scale_data(X_train, X_test, scaler=None):
        if scaler is None:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        logging.info("Data scaling complete using StandardScaler")
        return X_train_scaled, X_test_scaled, scaler

    @staticmethod
    def log_transform(X, columns):
        X_log = X.copy()
        for col in columns:
            X_log[col] = np.log1p(X_log[col])
            logging.info(f"Applied log transformation on column: {col}")
        return X_log

class LoggerUtils:
    @staticmethod
    def log_to_file(message, file_path):
        with open(file_path, 'a') as f:
            f.write(f"{datetime.now()} - {message}
")
        logging.info(f"Logged message to file: {file_path}")

    @staticmethod
    def setup_logger(logger_name, log_file, level=logging.INFO):
        logger = logging.getLogger(logger_name)
        handler = logging.FileHandler(log_file)        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.setLevel(level)
        logger.addHandler(handler)
        return logger

def main():
    FileUtils.create_directory('./logs')
    logger = LoggerUtils.setup_logger('AGI_Logger', './logs/agi_system.log')
    logger.info("AGI System Utilities Initialized")

if __name__ == "__main__":
    main()

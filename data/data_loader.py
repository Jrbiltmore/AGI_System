
import os
import pandas as pd
from sklearn.datasets import fetch_openml
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataLoader:
    def __init__(self, data_path, delimiter=','):
        self.data_path = data_path
        self.delimiter = delimiter

    def load_csv(self, file_name):
        file_path = os.path.join(self.data_path, file_name)
        if os.path.exists(file_path):
            data = pd.read_csv(file_path, delimiter=self.delimiter)
            logging.info(f"CSV data loaded successfully from {file_path}")
            return data
        else:
            logging.error(f"File {file_path} not found.")
            raise FileNotFoundError(f"File {file_path} not found.")

    def load_from_openml(self, dataset_id):
        try:
            data = fetch_openml(data_id=dataset_id)
            logging.info(f"Data loaded successfully from OpenML with dataset ID: {dataset_id}")
            return pd.DataFrame(data.data, columns=data.feature_names), pd.Series(data.target, name='target')
        except Exception as e:
            logging.error(f"Error loading data from OpenML: {str(e)}")
            raise

    def save_data(self, df, file_name):
        file_path = os.path.join(self.data_path, file_name)
        df.to_csv(file_path, index=False)
        logging.info(f"Data saved successfully to {file_path}")

    def list_files(self):
        try:
            files = os.listdir(self.data_path)
            logging.info(f"Files in {self.data_path}: {files}")
            return files
        except FileNotFoundError:
            logging.error(f"Directory {self.data_path} not found.")
            raise

def main():
    data_loader = DataLoader(data_path="/mnt/data/AGI_System/data/raw/")
    try:
        data_loader.list_files()
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()

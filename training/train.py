
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from core.main_model import AGIModel
from core.preprocessing import DataPreprocessor
from data.data_loader import DataLoader
from data.augmentation_tools import DataAugmentation
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_preprocess_data():
    data_loader = DataLoader(data_path='/mnt/data/AGI_System/data/raw/')
    df = data_loader.load_csv('training_data.csv')
    
    preprocessor = DataPreprocessor(strategy='mean', scaling_method='standard', apply_pca=False)
    X = df.drop('target', axis=1)
    y = df['target']
    X_processed = preprocessor.fit_transform(X)
    y_encoded = preprocessor.encode_labels(y)
    
    logging.info("Data loaded and preprocessed successfully")
    return X_processed, y_encoded

def augment_data(X, y):
    augmentor = DataAugmentation()
    X_noisy = augmentor.add_gaussian_noise(X)
    X_upsampled, y_upsampled = augmentor.oversample_minority_class(X_noisy, y, minority_class=1)
    logging.info("Data augmentation completed")
    return X_upsampled, y_upsampled

def train_model(X_train, y_train, X_val, y_val):
    model = AGIModel(input_shape=X_train.shape[1:], output_shape=len(np.unique(y_train)), learning_rate=1e-4)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('/mnt/data/AGI_System/training/checkpoints/best_model.h5', save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
    
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        batch_size=32,
        epochs=100,
        callbacks=[early_stopping, model_checkpoint, reduce_lr]
    )
    logging.info("Model training completed")
    return history, model

def main():
    X, y = load_and_preprocess_data()
    X_augmented, y_augmented = augment_data(X, y)
    X_train, X_val, y_train, y_val = DataPreprocessor.split_data(pd.DataFrame(X_augmented), 'target', test_size=0.2)
    history, model = train_model(X_train, y_train, X_val, y_val)
    model.save_model('/mnt/data/AGI_System/training/checkpoints/final_model.h5')
    logging.info("Final model saved successfully")

if __name__ == "__main__":
    main()

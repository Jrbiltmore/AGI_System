
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

class DataPreprocessor:
    def __init__(self, strategy='mean', scaling_method='standard', apply_pca=False, n_components=None):
        self.strategy = strategy
        self.scaling_method = scaling_method
        self.apply_pca = apply_pca
        self.n_components = n_components
        self.imputer = SimpleImputer(strategy=self.strategy)
        self.scaler = self._choose_scaler()
        self.pca = PCA(n_components=self.n_components) if self.apply_pca else None
        self.pipeline = self._build_pipeline()

    def _choose_scaler(self):
        if self.scaling_method == 'standard':
            return StandardScaler()
        elif self.scaling_method == 'minmax':
            return MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {self.scaling_method}")

    def _build_pipeline(self):
        steps = [('imputer', self.imputer), ('scaler', self.scaler)]
        if self.apply_pca:
            steps.append(('pca', self.pca))
        return Pipeline(steps)

    def fit(self, X):
        self.pipeline.fit(X)
        return self

    def transform(self, X):
        return self.pipeline.transform(X)

    def fit_transform(self, X):
        return self.pipeline.fit_transform(X)

    def encode_labels(self, y):
        label_encoder = LabelEncoder()
        return label_encoder.fit_transform(y)

    def inverse_transform_labels(self, encoded_y, label_encoder):
        return label_encoder.inverse_transform(encoded_y)

    def summary(self, X):
        if self.apply_pca:
            explained_variance = np.sum(self.pca.explained_variance_ratio_)
            return {
                'missing_values': np.sum(np.isnan(X)),
                'scaling_method': self.scaling_method,
                'pca_explained_variance': explained_variance,
                'n_components': self.n_components
            }
        else:
            return {
                'missing_values': np.sum(np.isnan(X)),
                'scaling_method': self.scaling_method
            }

def load_data(file_path, delimiter=','):
    return pd.read_csv(file_path, delimiter=delimiter)

def save_data(df, file_path):
    df.to_csv(file_path, index=False)

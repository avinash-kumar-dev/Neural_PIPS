import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
import joblib
import os


class FeaturePipeline:
    def __init__(self, variance_threshold=0.001):
        self.pipeline = Pipeline([
            ('variance', VarianceThreshold(threshold=variance_threshold)),
            ('scaler', MinMaxScaler()),
        ])
        self.fitted = False
        self.feature_names_in_ = None
        self.raw_features_ = None

    def fit_transform(self, X, feature_names=None):
        X_clean = self._clean(X)
        self.feature_names_in_ = feature_names or [f'f{i}' for i in range(X_clean.shape[1])]
        result = self.pipeline.fit_transform(X_clean)
        self.fitted = True
        kept = self.pipeline.named_steps['variance'].get_support()
        self.feature_names_out_ = [n for n, k in zip(self.feature_names_in_, kept) if k]
        self.raw_features_ = X_clean[:, kept]
        return result

    def transform(self, X):
        if not self.fitted:
            raise ValueError("Pipeline not fitted. Call fit_transform first.")
        X_clean = self._clean(X)
        return self.pipeline.transform(X_clean)

    def fit_transform_df(self, X_df):
        names = list(X_df.columns)
        self.feature_names_in_ = names
        X_np = self.fit_transform(X_df, names)
        self.raw_features_ = X_df.values.copy()
        return pd.DataFrame(X_np, columns=self.feature_names_out_, index=X_df.index)

    def transform_df(self, X_df):
        X_np = self.transform(X_df)
        return pd.DataFrame(X_np, columns=self.feature_names_out_, index=X_df.index)

    def get_raw_features_df(self):
        if self.raw_features_ is None:
            raise ValueError("No raw features available. Call fit_transform_df first.")
        return pd.DataFrame(self.raw_features_, columns=self.feature_names_in_)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)

    def _clean(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values.copy()
        else:
            X = X.copy()
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X

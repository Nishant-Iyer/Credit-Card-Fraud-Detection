import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler
from typing import List, Optional

class CyclicalTimeEncoder(BaseEstimator, TransformerMixin):
    """
    Transforms a time column representing seconds elapsed into cyclical sine/cosine features
    to capture time-of-day periodic patterns.
    """
    def __init__(self, time_col: str = "Time", period: float = 86400.0, drop_original: bool = True):
        self.time_col = time_col
        self.period = period
        self.drop_original = drop_original

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "CyclicalTimeEncoder":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        if self.time_col in X_out.columns:
            # Convert seconds to radians
            time_rad = 2 * np.pi * X_out[self.time_col] / self.period
            X_out[f"{self.time_col}_sin"] = np.sin(time_rad)
            X_out[f"{self.time_col}_cos"] = np.cos(time_rad)
            
            if self.drop_original:
                X_out = X_out.drop(columns=[self.time_col])
        return X_out


class DataFrameRobustScaler(BaseEstimator, TransformerMixin):
    """
    Applies RobustScaler to specific columns of a pandas DataFrame, 
    retaining column names and DataFrame format.
    """
    def __init__(self, scale_cols: List[str]):
        self.scale_cols = scale_cols
        self.scaler = RobustScaler()

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "DataFrameRobustScaler":
        missing_cols = [c for c in self.scale_cols if c not in X.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in DataFrame for scaling: {missing_cols}")
        
        self.scaler.fit(X[self.scale_cols])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        X_out[self.scale_cols] = self.scaler.transform(X_out[self.scale_cols])
        return X_out


class ColumnDropper(BaseEstimator, TransformerMixin):
    """
    Drops specific columns from a pandas DataFrame if they are present.
    Useful for removing columns like 'Time' or 'Class' from the feature set.
    """
    def __init__(self, drop_cols: List[str]):
        self.drop_cols = drop_cols

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "ColumnDropper":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        existing_cols = [c for c in self.drop_cols if c in X_out.columns]
        if existing_cols:
            X_out = X_out.drop(columns=existing_cols)
        return X_out


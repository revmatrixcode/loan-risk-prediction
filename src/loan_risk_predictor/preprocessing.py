"""
Data preprocessing module for loan risk prediction
"""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import joblib
from typing import Tuple, Optional

from .config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN


class DataPreprocessor:
    """
    Handles data preprocessing for loan risk prediction
    """
    
    def __init__(self):
        self.preprocessor = None
        self.is_fitted = False
        
    def create_preprocessor(self) -> ColumnTransformer:
        """
        Create the preprocessing pipeline
        """
        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(
                handle_unknown='ignore',
                sparse_output=False,
                drop='first'
            ))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, NUMERICAL_FEATURES),
                ('cat', categorical_transformer, CATEGORICAL_FEATURES)
            ],
            remainder='drop',
            verbose_feature_names_out=False
        )
        
        return preprocessor
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'DataPreprocessor':
        """
        Fit the preprocessor to data
        """
        self.preprocessor = self.create_preprocessor()
        self.preprocessor.fit(X, y)
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform data using fitted preprocessor
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transforming data")
        
        return self.preprocessor.transform(X)
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> np.ndarray:
        """
        Fit and transform data
        """
        return self.fit(X, y).transform(X)
    
    def get_feature_names(self) -> list:
        """
        Get feature names after preprocessing
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted first")
        
        return self.preprocessor.get_feature_names_out().tolist()
    
    def save(self, filepath: str) -> None:
        """
        Save preprocessor to file
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted preprocessor")
        
        joblib.dump(self.preprocessor, filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'DataPreprocessor':
        """
        Load preprocessor from file
        """
        instance = cls()
        instance.preprocessor = joblib.load(filepath)
        instance.is_fitted = True
        return instance


def prepare_training_data(df: pd.DataFrame, target_col: str = TARGET_COLUMN) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare data for training by separating features and target
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    return X, y

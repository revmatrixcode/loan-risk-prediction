"""
Model training and evaluation module
"""

import pandas as pd
import numpy as np
import joblib
from typing import Tuple, Dict, Any, Optional
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

from .preprocessing import DataPreprocessor, prepare_training_data
from .feature_engineering import FeatureEngineer
from .config import MODEL_PARAMS, DEFAULT_THRESHOLD, RISK_LEVELS


class LoanRiskModel:
    """
    Main model class for loan risk prediction
    """
    
    def __init__(self, model_params: Dict[str, Any] = None):
        """
        Initialize the model with parameters
        """
        self.model_params = model_params or MODEL_PARAMS.copy()
        self.model = XGBClassifier(**self.model_params)
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.is_trained = False
        
    def prepare_data(self, df: pd.DataFrame, target_col: str, 
                     test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """
        Prepare data with feature engineering and train-test split
        """
        # Apply feature engineering
        df_engineered = self.feature_engineer.engineer_features(df, target_col)
        
        # Split features and target
        X, y = prepare_training_data(df_engineered, target_col)
        
        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              scale_pos_weight: float = None) -> 'LoanRiskModel':
        """
        Train the model
        """
        # Calculate scale_pos_weight if not provided
        if scale_pos_weight is None:
            scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
        
        # Update model parameters
        self.model.set_params(scale_pos_weight=scale_pos_weight)
        
        # Preprocess the data
        X_train_processed = self.preprocessor.fit_transform(X_train, y_train)
        
        # Train the model
        self.model.fit(X_train_processed, y_train)
        self.is_trained = True
        
        return self
    
    def predict(self, X: pd.DataFrame, threshold: float = DEFAULT_THRESHOLD) -> np.ndarray:
        """
        Make binary predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_processed = self.preprocessor.transform(X)
        probabilities = self.model.predict_proba(X_processed)[:, 1]
        predictions = (probabilities >= threshold).astype(int)
        
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_processed = self.preprocessor.transform(X)
        probabilities = self.model.predict_proba(X_processed)
        
        return probabilities
    
    def predict_with_details(self, X: pd.DataFrame, 
                            threshold: float = DEFAULT_THRESHOLD) -> pd.DataFrame:
        """
        Make predictions with detailed information
        """
        probabilities = self.predict_proba(X)[:, 1]
        predictions = (probabilities >= threshold).astype(int)
        
        # Create detailed results
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            # Determine risk level
            if prob < RISK_LEVELS['LOW'][1]:
                risk_level = 'LOW'
            elif prob < RISK_LEVELS['HIGH'][0]:
                risk_level = 'MEDIUM'
            else:
                risk_level = 'HIGH'
            
            # Determine decision
            decision = 'APPROVE' if pred == 0 else 'REJECT'
            
            # Determine confidence
            if prob < 0.3 or prob > 0.7:
                confidence = 'HIGH'
            else:
                confidence = 'MEDIUM'
            
            results.append({
                'prediction': int(pred),
                'probability': float(prob),
                'risk_level': risk_level,
                'decision': decision,
                'confidence': confidence
            })
        
        return pd.DataFrame(results)
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series, 
                 threshold: float = DEFAULT_THRESHOLD) -> Dict[str, float]:
        """
        Evaluate model performance
        """
        predictions = self.predict(X_test, threshold)
        probabilities = self.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions),
            'recall': recall_score(y_test, predictions),
            'f1_score': f1_score(y_test, predictions),
            'roc_auc': roc_auc_score(y_test, probabilities)
        }
        
        return metrics
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, 
                      cv: int = 5, scoring: str = 'roc_auc') -> Dict[str, Any]:
        """
        Perform cross-validation
        """
        X_processed = self.preprocessor.fit_transform(X, y)
        
        cv_scores = cross_val_score(
            self.model, X_processed, y,
            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
            scoring=scoring,
            n_jobs=-1
        )
        
        return {
            'scores': cv_scores.tolist(),
            'mean': float(cv_scores.mean()),
            'std': float(cv_scores.std())
        }
    
    def save(self, model_path: str, preprocessor_path: str) -> None:
        """
        Save model and preprocessor
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        joblib.dump(self.model, model_path)
        self.preprocessor.save(preprocessor_path)
    
    @classmethod
    def load(cls, model_path: str, preprocessor_path: str) -> 'LoanRiskModel':
        """
        Load model and preprocessor
        """
        instance = cls()
        instance.model = joblib.load(model_path)
        instance.preprocessor = DataPreprocessor.load(preprocessor_path)
        instance.is_trained = True
        
        return instance
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")
        
        feature_names = self.preprocessor.get_feature_names()
        importances = self.model.feature_importances_
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df

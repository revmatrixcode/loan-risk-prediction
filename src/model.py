import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

class LoanRiskPredictor:
    """Main class for loan risk prediction."""
    
    def __init__(self, model_path=None):
        self.model = None
        self.pipeline = None
        
        if model_path:
            self.load_model(model_path)
    
    def create_pipeline(self, scale_pos_weight=7.1):
        """Create the ML pipeline."""
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        
        # Define feature types
        numerical_features = ['Income', 'Age', 'Experience', 
                             'CURRENT_JOB_YRS', 'CURRENT_HOUSE_YRS']
        categorical_features = ['Married/Single', 'House_Ownership', 
                               'Car_Ownership', 'Profession', 'CITY', 'STATE']
        
        # Create preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), 
                 categorical_features)
            ]
        )
        
        # Create pipeline
        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss'
            ))
        ])
        
        return self.pipeline
    
    def train(self, X, y, scale_pos_weight=7.1):
        """Train the model."""
        if self.pipeline is None:
            self.create_pipeline(scale_pos_weight)
        
        self.pipeline.fit(X, y)
        self.model = self.pipeline.named_steps['classifier']
        
        return self
    
    def predict(self, X):
        """Make predictions."""
        if self.pipeline is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.pipeline.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities."""
        if self.pipeline is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.pipeline.predict_proba(X)[:, 1]
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance."""
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        metrics = {
            'roc_auc': roc_auc_score(y_test, y_proba),
            'accuracy': accuracy_score(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, 
                                                         output_dict=True)
        }
        
        return metrics
    
    def save_model(self, filepath):
        """Save the trained model."""
        if self.pipeline is None:
            raise ValueError("No model to save. Train the model first.")
        
        joblib.dump(self.pipeline, filepath)
    
    def load_model(self, filepath):
        """Load a trained model."""
        self.pipeline = joblib.load(filepath)
        self.model = self.pipeline.named_steps['classifier']

#!/usr/bin/env python3
"""
Script to train and save the loan risk prediction model
"""

import sys
import os
import pandas as pd
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from loan_risk_predictor.model import LoanRiskModel
from loan_risk_predictor.feature_engineering import FeatureEngineer
from loan_risk_predictor.config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES

def main():
    print("🚀 Training Loan Risk Prediction Model")
    print("=" * 50)
    
    # Create sample training data
    print("\n1. Creating sample training data...")
    np.random.seed(42)
    
    # Generate realistic sample data
    data = {
        'Income': np.random.randint(500000, 10000000, 1000),
        'Age': np.random.randint(20, 70, 1000),
        'Experience': np.random.randint(0, 50, 1000),
        'Married/Single': np.random.choice(['single', 'married'], 1000),
        'House_Ownership': np.random.choice(['rented', 'owned', 'norent_noown'], 1000),
        'Car_Ownership': np.random.choice(['yes', 'no'], 1000),
        'Profession': np.random.choice(['Engineer', 'Doctor', 'Teacher', 'Lawyer', 'Accountant'], 1000),
        'CITY': np.random.choice(['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Hyderabad'], 1000),
        'STATE': np.random.choice(['Maharashtra', 'Delhi', 'Karnataka', 'Tamil Nadu', 'Telangana'], 1000),
        'CURRENT_JOB_YRS': np.random.randint(0, 20, 1000),
        'CURRENT_HOUSE_YRS': np.random.randint(0, 30, 1000),
        'Risk_Flag': np.random.choice([0, 1], 1000, p=[0.88, 0.12])
    }
    
    df = pd.DataFrame(data)
    print(f"   Created {len(df)} samples")
    print(f"   Risk distribution: {df['Risk_Flag'].value_counts().to_dict()}")
    
    # Create and train model
    print("\n2. Training model...")
    model = LoanRiskModel()
    X_train, X_test, y_train, y_test = model.prepare_data(df, 'Risk_Flag', test_size=0.2)
    
    model.train(X_train, y_train)
    print(f"   Model trained successfully")
    
    # Evaluate model
    print("\n3. Evaluating model...")
    metrics = model.evaluate(X_test, y_test)
    print(f"   ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"   Accuracy: {metrics['accuracy']:.4f}")
    print(f"   F1-Score: {metrics['f1_score']:.4f}")
    
    # Cross-validation
    print("\n4. Cross-validation...")
    cv_results = model.cross_validate(X_train, y_train, cv=3)
    print(f"   Mean ROC-AUC: {cv_results['mean']:.4f}")
    print(f"   Std ROC-AUC: {cv_results['std']:.4f}")
    
    # Save model artifacts
    print("\n5. Saving model artifacts...")
    artifacts_dir = 'model_artifacts'
    os.makedirs(artifacts_dir, exist_ok=True)
    
    model_path = os.path.join(artifacts_dir, 'model.pkl')
    preprocessor_path = os.path.join(artifacts_dir, 'preprocessor.pkl')
    
    model.save(model_path, preprocessor_path)
    
    # Create metadata
    metadata = {
        'model_info': {
            'name': 'Loan Risk Prediction Model',
            'version': '1.0.0',
            'type': 'XGBoost Classifier',
            'training_date': datetime.now().isoformat(),
            'training_samples': len(X_train),
            'features': {
                'numerical': NUMERICAL_FEATURES,
                'categorical': CATEGORICAL_FEATURES,
                'engineered': FeatureEngineer().get_feature_types()
            }
        },
        'performance': {
            'test_set': metrics,
            'cross_validation': cv_results
        },
        'parameters': model.model.get_params(),
        'feature_importance': model.get_feature_importance().head(10).to_dict('records')
    }
    
    metadata_path = os.path.join(artifacts_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print(f"\n✅ Model artifacts saved to '{artifacts_dir}/'")
    print(f"   - model.pkl: Trained XGBoost model")
    print(f"   - preprocessor.pkl: Data preprocessor")
    print(f"   - metadata.json: Model metadata")
    
    # Test loading and prediction
    print("\n6. Testing model loading...")
    loaded_model = LoanRiskModel.load(model_path, preprocessor_path)
    test_predictions = loaded_model.predict(X_test.head(5))
    print(f"   Sample predictions: {test_predictions}")
    
    print("\n🎉 Model training complete!")

if __name__ == "__main__":
    import numpy as np
    main()
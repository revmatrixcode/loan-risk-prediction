import pytest
import pandas as pd
import numpy as np
from src.loan_risk_predictor.model import LoanRiskModel

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'Income': np.random.randint(500000, 10000000, n_samples),
        'Age': np.random.randint(20, 70, n_samples),
        'Experience': np.random.randint(0, 50, n_samples),
        'Married/Single': np.random.choice(['single', 'married'], n_samples),
        'House_Ownership': np.random.choice(['rented', 'owned'], n_samples),
        'Car_Ownership': np.random.choice(['yes', 'no'], n_samples),
        'Profession': np.random.choice(['Engineer', 'Doctor'], n_samples),
        'CITY': np.random.choice(['CityA', 'CityB'], n_samples),
        'STATE': np.random.choice(['StateA', 'StateB'], n_samples),
        'CURRENT_JOB_YRS': np.random.randint(0, 20, n_samples),
        'CURRENT_HOUSE_YRS': np.random.randint(0, 30, n_samples),
        'Risk_Flag': np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
    }
    
    return pd.DataFrame(data)

def test_model_initialization():
    """Test model initialization"""
    model = LoanRiskModel()
    assert model.model is not None
    assert not model.is_trained

def test_prepare_data(sample_data):
    """Test data preparation"""
    model = LoanRiskModel()
    X_train, X_test, y_train, y_test = model.prepare_data(
        sample_data, 'Risk_Flag', test_size=0.3
    )
    
    assert len(X_train) + len(X_test) == len(sample_data)
    assert len(y_train) + len(y_test) == len(sample_data)

def test_model_training(sample_data):
    """Test model training"""
    model = LoanRiskModel()
    X_train, X_test, y_train, y_test = model.prepare_data(
        sample_data, 'Risk_Flag', test_size=0.3
    )
    
    model.train(X_train, y_train)
    assert model.is_trained

def test_predictions(sample_data):
    """Test model predictions"""
    model = LoanRiskModel()
    X_train, X_test, y_train, y_test = model.prepare_data(
        sample_data, 'Risk_Flag', test_size=0.3
    )
    
    model.train(X_train, y_train)
    predictions = model.predict(X_test)
    
    assert len(predictions) == len(X_test)
    assert predictions.dtype == np.int64

def test_feature_importance(sample_data):
    """Test feature importance extraction"""
    model = LoanRiskModel()
    X_train, X_test, y_train, y_test = model.prepare_data(
        sample_data, 'Risk_Flag', test_size=0.3
    )
    
    model.train(X_train, y_train)
    importance_df = model.get_feature_importance()
    
    assert isinstance(importance_df, pd.DataFrame)
    assert 'feature' in importance_df.columns
    assert 'importance' in importance_df.columns

"""
Test configuration and shared fixtures
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

@pytest.fixture(scope="session")
def test_data_path():
    """Get path to test data"""
    return os.path.join(os.path.dirname(__file__), '../examples/sample_data.csv')

@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing"""
    np.random.seed(42)
    
    data = {
        'Income': np.random.randint(500000, 10000000, 50),
        'Age': np.random.randint(20, 70, 50),
        'Experience': np.random.randint(0, 50, 50),
        'Married/Single': np.random.choice(['single', 'married'], 50),
        'House_Ownership': np.random.choice(['rented', 'owned', 'norent_noown'], 50),
        'Car_Ownership': np.random.choice(['yes', 'no'], 50),
        'Profession': np.random.choice(['Engineer', 'Doctor', 'Teacher', 'Lawyer'], 50),
        'CITY': np.random.choice(['Mumbai', 'Delhi', 'Bangalore', 'Chennai'], 50),
        'STATE': np.random.choice(['Maharashtra', 'Delhi', 'Karnataka', 'Tamil Nadu'], 50),
        'CURRENT_JOB_YRS': np.random.randint(0, 20, 50),
        'CURRENT_HOUSE_YRS': np.random.randint(0, 30, 50),
        'Risk_Flag': np.random.choice([0, 1], 50, p=[0.88, 0.12])
    }
    
    return pd.DataFrame(data)

@pytest.fixture
def trained_model(sample_dataframe):
    """Create a trained model for testing"""
    from src.loan_risk_predictor.model import LoanRiskModel
    
    model = LoanRiskModel()
    X_train, X_test, y_train, y_test = model.prepare_data(
        sample_dataframe, 'Risk_Flag', test_size=0.3
    )
    model.train(X_train, y_train)
    
    return model, X_test, y_test

@pytest.fixture
def preprocessor():
    """Create a preprocessor instance"""
    from src.loan_risk_predictor.preprocessing import DataPreprocessor
    return DataPreprocessor()

@pytest.fixture
def feature_engineer():
    """Create a feature engineer instance"""
    from src.loan_risk_predictor.feature_engineering import FeatureEngineer
    return FeatureEngineer()

@pytest.fixture
def sample_applicant():
    """Sample applicant data for testing"""
    return {
        'Income': 1303834,
        'Age': 23,
        'Experience': 3,
        'Married/Single': 'single',
        'House_Ownership': 'rented',
        'Car_Ownership': 'no',
        'Profession': 'Mechanical_engineer',
        'CITY': 'Rewa',
        'STATE': 'Madhya_Pradesh',
        'CURRENT_JOB_YRS': 3,
        'CURRENT_HOUSE_YRS': 13
    }

@pytest.fixture
def sample_applicants():
    """Multiple sample applicants"""
    return [
        {
            'Income': 1303834,
            'Age': 23,
            'Experience': 3,
            'Married/Single': 'single',
            'House_Ownership': 'rented',
            'Car_Ownership': 'no',
            'Profession': 'Mechanical_engineer',
            'CITY': 'Rewa',
            'STATE': 'Madhya_Pradesh',
            'CURRENT_JOB_YRS': 3,
            'CURRENT_HOUSE_YRS': 13
        },
        {
            'Income': 7574516,
            'Age': 40,
            'Experience': 10,
            'Married/Single': 'single',
            'House_Ownership': 'rented',
            'Car_Ownership': 'no',
            'Profession': 'Software_Developer',
            'CITY': 'Parbhani',
            'STATE': 'Maharashtra',
            'CURRENT_JOB_YRS': 9,
            'CURRENT_HOUSE_YRS': 13
        }
    ]

# Test data generators
@pytest.fixture
def generate_test_data():
    """Generate test data with specified characteristics"""
    def _generate(n_samples=100, risk_ratio=0.12):
        np.random.seed(42)
        
        data = {
            'Income': np.random.lognormal(14, 0.5, n_samples).astype(int),
            'Age': np.random.randint(20, 70, n_samples),
            'Experience': np.random.randint(0, 50, n_samples),
            'Married/Single': np.random.choice(['single', 'married'], n_samples),
            'House_Ownership': np.random.choice(['rented', 'owned', 'norent_noown'], n_samples),
            'Car_Ownership': np.random.choice(['yes', 'no'], n_samples),
            'Profession': np.random.choice(['Engineer', 'Doctor', 'Teacher', 'Lawyer', 'Accountant'], n_samples),
            'CITY': np.random.choice(['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Hyderabad'], n_samples),
            'STATE': np.random.choice(['Maharashtra', 'Delhi', 'Karnataka', 'Tamil Nadu', 'Telangana'], n_samples),
            'CURRENT_JOB_YRS': np.random.randint(0, 20, n_samples),
            'CURRENT_HOUSE_YRS': np.random.randint(0, 30, n_samples),
            'Risk_Flag': np.random.choice([0, 1], n_samples, p=[1-risk_ratio, risk_ratio])
        }
        
        return pd.DataFrame(data)
    
    return _generate

# Configuration for tests
def pytest_configure(config):
    """Configure pytest"""
    # Register custom markers
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")

def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    for item in items:
        # Mark tests based on file name
        if "test_" in item.nodeid and "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        elif "test_" in item.nodeid:
            item.add_marker(pytest.mark.unit)
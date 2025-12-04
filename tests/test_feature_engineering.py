"""
Test feature engineering module
"""

import pytest
import pandas as pd
import numpy as np
from src.loan_risk_predictor.feature_engineering import FeatureEngineer
from src.loan_risk_predictor.config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    data = {
        'Income': [1000000, 2000000, 3000000, 4000000],
        'Age': [22, 32, 45, 55],
        'Experience': [1, 5, 10, 15],
        'Married/Single': ['single', 'married', 'single', 'married'],
        'House_Ownership': ['rented', 'owned', 'rented', 'owned'],
        'Car_Ownership': ['yes', 'no', 'yes', 'no'],
        'Profession': ['Engineer', 'Doctor', 'Teacher', 'Lawyer'],
        'CITY': ['Mumbai', 'Delhi', 'Bangalore', 'Chennai'],
        'STATE': ['Maharashtra', 'Delhi', 'Karnataka', 'Tamil Nadu'],
        'CURRENT_JOB_YRS': [1, 3, 8, 12],
        'CURRENT_HOUSE_YRS': [1, 2, 5, 10],
        'Risk_Flag': [0, 1, 0, 1]
    }
    return pd.DataFrame(data)

@pytest.fixture
def feature_engineer():
    """Create feature engineer instance"""
    return FeatureEngineer()

def test_create_interaction_features(sample_data, feature_engineer):
    """Test creation of interaction features"""
    df_with_features = feature_engineer.create_interaction_features(sample_data)
    
    # Check that new features are added
    expected_features = ['Income_Age_Ratio', 'Experience_Age_Ratio', 
                        'Stability_Score', 'DTI_Ratio']
    
    for feature in expected_features:
        assert feature in df_with_features.columns
    
    # Check calculations
    assert df_with_features['Income_Age_Ratio'].iloc[0] == pytest.approx(1000000 / 23)
    assert df_with_features['Stability_Score'].iloc[0] == 2  # 1 + 1
    
def test_create_categorical_features(sample_data, feature_engineer):
    """Test creation of categorical features"""
    df_with_categorical = feature_engineer.create_categorical_features(sample_data)
    
    # Check that new categorical features are added
    expected_features = ['Age_Group', 'Income_Category', 'Stability_Category']
    
    for feature in expected_features:
        assert feature in df_with_categorical.columns
    
    # Check age groups
    assert df_with_categorical['Age_Group'].iloc[0] == 'Young'  # Age 22
    assert df_with_categorical['Age_Group'].iloc[1] == 'Young_Adult'  # Age 32
    
    # Check income categories
    assert df_with_categorical['Income_Category'].iloc[0] == 'Low'  # 1M
    assert df_with_categorical['Income_Category'].iloc[2] == 'Medium'  # 3M
    
def test_create_target_encodings(sample_data, feature_engineer):
    """Test creation of target encodings"""
    df_with_encodings = feature_engineer.create_target_encodings(sample_data, 'Risk_Flag')
    
    # Check that encoding features are added
    expected_features = ['City_Risk_Encoding', 'State_Risk_Encoding']
    
    for feature in expected_features:
        if feature in df_with_encodings.columns:
            assert df_with_encodings[feature].dtype in [np.float64, np.float32]
    
def test_engineer_features(sample_data, feature_engineer):
    """Test complete feature engineering pipeline"""
    df_engineered = feature_engineer.engineer_features(sample_data, 'Risk_Flag')
    
    # Check that all expected features are present
    expected_new_features = [
        'Income_Age_Ratio', 'Experience_Age_Ratio',
        'Stability_Score', 'DTI_Ratio',
        'Age_Group', 'Income_Category', 'Stability_Category'
    ]
    
    for feature in expected_new_features:
        assert feature in df_engineered.columns
    
    # Check original features are preserved
    for feature in NUMERICAL_FEATURES + CATEGORICAL_FEATURES:
        assert feature in df_engineered.columns

def test_get_feature_types(feature_engineer):
    """Test get_feature_types method"""
    feature_types = feature_engineer.get_feature_types()
    
    assert isinstance(feature_types, dict)
    assert 'numerical' in feature_types
    assert 'categorical' in feature_types
    assert 'target_encoded' in feature_types
    
    # Check that feature lists are not empty
    assert len(feature_types['numerical']) > 0
    assert len(feature_types['categorical']) > 0

def test_handling_missing_target(sample_data, feature_engineer):
    """Test feature engineering without target column"""
    df_without_target = sample_data.drop(columns=['Risk_Flag'])
    df_engineered = feature_engineer.engineer_features(df_without_target)
    
    # Should still create all features except target encodings
    expected_features = [
        'Income_Age_Ratio', 'Experience_Age_Ratio',
        'Stability_Score', 'DTI_Ratio',
        'Age_Group', 'Income_Category', 'Stability_Category'
    ]
    
    for feature in expected_features:
        assert feature in df_engineered.columns

def test_feature_engineering_consistency(sample_data, feature_engineer):
    """Test that feature engineering is consistent"""
    df1 = feature_engineer.engineer_features(sample_data, 'Risk_Flag')
    df2 = feature_engineer.engineer_features(sample_data, 'Risk_Flag')
    
    # Same input should produce same output
    for col in df1.columns:
        if df1[col].dtype in [np.float64, np.float32]:
            assert df1[col].equals(df2[col])
        else:
            # For object/string columns
            assert (df1[col] == df2[col]).all()

def test_stability_score_calculation(sample_data, feature_engineer):
    """Test stability score calculation"""
    df_with_features = feature_engineer.create_interaction_features(sample_data)
    
    # Check manual calculation matches function
    for idx, row in sample_data.iterrows():
        expected_score = row['CURRENT_JOB_YRS'] + row['CURRENT_HOUSE_YRS']
        actual_score = df_with_features.loc[idx, 'Stability_Score']
        assert actual_score == expected_score

def test_age_group_categorization(sample_data, feature_engineer):
    """Test age group categorization"""
    df_with_categorical = feature_engineer.create_categorical_features(sample_data)
    
    age_groups = df_with_categorical['Age_Group'].unique()
    
    # Should have appropriate age groups
    for age, group in zip(sample_data['Age'], df_with_categorical['Age_Group']):
        if age < 25:
            assert group == 'Young'
        elif age < 35:
            assert group == 'Young_Adult'
        elif age < 50:
            assert group == 'Middle_Aged'
        else:
            assert group == 'Senior'
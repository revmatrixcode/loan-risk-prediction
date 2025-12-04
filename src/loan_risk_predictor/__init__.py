"""
Loan Risk Predictor - A machine learning package for predicting loan default risk
"""

__version__ = "1.0.0"
__author__ = "Shanujan Suresh"
__email__ = "shanujansh@gmail.com"

from .model import LoanRiskModel
from .predict import predict_risk, predict_batch
from .preprocessing import DataPreprocessor
from .feature_engineering import FeatureEngineer

__all__ = [
    "LoanRiskModel",
    "predict_risk",
    "predict_batch",
    "DataPreprocessor",
    "FeatureEngineer",
]

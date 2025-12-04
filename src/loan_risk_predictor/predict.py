"""
Prediction utilities for easy API usage
"""

import pandas as pd
import numpy as np
from typing import Union, List, Dict, Any
import joblib
from pathlib import Path

from .model import LoanRiskModel
from .config import DEFAULT_MODEL_PATH, DEFAULT_PREPROCESSOR_PATH


def load_model(model_path: str = DEFAULT_MODEL_PATH,
               preprocessor_path: str = DEFAULT_PREPROCESSOR_PATH) -> LoanRiskModel:
    """
    Load trained model and preprocessor
    
    Args:
        model_path: Path to saved model
        preprocessor_path: Path to saved preprocessor
    
    Returns:
        Loaded LoanRiskModel instance
    """
    return LoanRiskModel.load(model_path, preprocessor_path)


def predict_risk(input_data: Union[pd.DataFrame, Dict[str, Any], List[Dict]],
                 model_path: str = DEFAULT_MODEL_PATH,
                 preprocessor_path: str = DEFAULT_PREPROCESSOR_PATH,
                 threshold: float = 0.5) -> Union[Dict, List[Dict]]:
    """
    Predict loan risk for input data
    
    Args:
        input_data: Input data as DataFrame, dict, or list of dicts
        model_path: Path to saved model
        preprocessor_path: Path to saved preprocessor
        threshold: Decision threshold
    
    Returns:
        Predictions with details
    """
    # Load model
    model = load_model(model_path, preprocessor_path)
    
    # Convert input to DataFrame
    if isinstance(input_data, dict):
        input_data = pd.DataFrame([input_data])
    elif isinstance(input_data, list):
        input_data = pd.DataFrame(input_data)
    
    # Make predictions
    predictions_df = model.predict_with_details(input_data, threshold)
    
    # Convert to appropriate output format
    if len(predictions_df) == 1:
        return predictions_df.iloc[0].to_dict()
    else:
        return predictions_df.to_dict('records')


def predict_batch(csv_path: str,
                  model_path: str = DEFAULT_MODEL_PATH,
                  preprocessor_path: str = DEFAULT_PREPROCESSOR_PATH,
                  threshold: float = 0.5,
                  output_path: str = None) -> pd.DataFrame:
    """
    Predict loan risk for a batch of data from CSV
    
    Args:
        csv_path: Path to input CSV file
        model_path: Path to saved model
        preprocessor_path: Path to saved preprocessor
        threshold: Decision threshold
        output_path: Path to save predictions (optional)
    
    Returns:
        DataFrame with predictions
    """
    # Load data
    data = pd.read_csv(csv_path)
    
    # Load model
    model = load_model(model_path, preprocessor_path)
    
    # Make predictions
    predictions_df = model.predict_with_details(data, threshold)
    
    # Add predictions to original data
    result_df = pd.concat([data.reset_index(drop=True), predictions_df], axis=1)
    
    # Save if output path provided
    if output_path:
        result_df.to_csv(output_path, index=False)
    
    return result_df


def get_model_info(model_path: str = DEFAULT_MODEL_PATH) -> Dict[str, Any]:
    """
    Get information about the trained model
    
    Args:
        model_path: Path to saved model
    
    Returns:
        Dictionary with model information
    """
    model = joblib.load(model_path)
    
    info = {
        'model_type': type(model).__name__,
        'n_estimators': model.n_estimators if hasattr(model, 'n_estimators') else None,
        'learning_rate': model.learning_rate if hasattr(model, 'learning_rate') else None,
        'max_depth': model.max_depth if hasattr(model, 'max_depth') else None,
        'feature_importances': model.feature_importances_.tolist() 
                              if hasattr(model, 'feature_importances_') else None
    }
    
    return info

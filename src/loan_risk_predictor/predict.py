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

def main():
    """
    Command-line interface for loan risk prediction
    """
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(
        description='Predict loan risk from command line'
    )
    parser.add_argument(
        '--input', 
        type=str, 
        required=True,
        help='Input CSV file path'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='predictions.csv',
        help='Output file path (default: predictions.csv)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Decision threshold (default: 0.5)'
    )
    
    args = parser.parse_args()
    
    try:
        # Make predictions
        results = predict_batch(
            args.input,
            output_path=args.output,
            threshold=args.threshold
        )
        
        print(f"Predictions saved to {args.output}")
        print(f"Total applications processed: {len(results)}")
        
        # Show summary
        approved = len(results[results['decision'] == 'APPROVE'])
        rejected = len(results[results['decision'] == 'REJECT'])
        
        print(f"\n📈 Summary:")
        print(f"  Approved: {approved} ({approved/len(results):.1%})")
        print(f"  Rejected: {rejected} ({rejected/len(results):.1%})")
        
        # Risk level distribution
        print(f"\n Risk Levels:")
        for level in ['LOW', 'MEDIUM', 'HIGH']:
            count = len(results[results['risk_level'] == level])
            print(f"  {level}: {count} ({count/len(results):.1%})")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
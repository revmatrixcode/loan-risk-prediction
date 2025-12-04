"""
Example: API usage with loan risk predictor
"""

from loan_risk_predictor import predict_risk, get_model_info

def main():
    # Single prediction
    print("Single Application Prediction")
    print("=" * 60)
    
    applicant_data = {
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
    
    result = predict_risk(applicant_data)
    
    print(f"Applicant Details:")
    for key, value in applicant_data.items():
        print(f"  {key}: {value}")
    
    print(f"\nRisk Assessment:")
    print(f"  Risk Level: {result['risk_level']}")
    print(f"  Decision: {result['decision']}")
    print(f"  Confidence: {result['confidence']}")
    print(f"  Probability: {result['probability']:.2%}")
    
    # Get model info
    print("\n" + "=" * 60)
    print("Model Information")
    print("=" * 60)
    
    model_info = get_model_info()
    print(f"Model Type: {model_info['model_type']}")
    print(f"Number of Trees: {model_info['n_estimators']}")
    print(f"Learning Rate: {model_info['learning_rate']}")
    print(f"Max Depth: {model_info['max_depth']}")

if __name__ == "__main__":
    main()

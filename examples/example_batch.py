"""
Example: Batch prediction with loan risk predictor
"""

import pandas as pd
from loan_risk_predictor import predict_batch

def main():
    # Create sample data
    data = {
        'Income': [1303834, 7574516, 3991815],
        'Age': [23, 40, 66],
        'Experience': [3, 10, 4],
        'Married/Single': ['single', 'single', 'married'],
        'House_Ownership': ['rented', 'rented', 'rented'],
        'Car_Ownership': ['no', 'no', 'no'],
        'Profession': ['Mechanical_engineer', 'Software_Developer', 'Technical_writer'],
        'CITY': ['Rewa', 'Parbhani', 'Alappuzha'],
        'STATE': ['Madhya_Pradesh', 'Maharashtra', 'Kerala'],
        'CURRENT_JOB_YRS': [3, 9, 4],
        'CURRENT_HOUSE_YRS': [13, 13, 10]
    }
    
    df = pd.DataFrame(data)
    df.to_csv('sample_loans.csv', index=False)
    
    # Make predictions
    print("Making batch predictions...")
    results = predict_batch(
        'sample_loans.csv',
        output_path='predictions.csv',
        threshold=0.5
    )
    
    print(f"\nPredictions saved to predictions.csv")
    print(f"Total applications processed: {len(results)}")
    
    # Display results
    print("\nPrediction Summary:")
    print("=" * 60)
    for i, row in results.iterrows():
        print(f"Application {i+1}:")
        print(f"  Risk Level: {row['risk_level']}")
        print(f"  Decision: {row['decision']}")
        print(f"  Confidence: {row['confidence']}")
        print(f"  Probability: {row['probability']:.2%}")
        print()

if __name__ == "__main__":
    main()

"""
Utility functions for loan risk prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import json
from datetime import datetime
from typing import Dict, Any, List, Tuple
import warnings
warnings.filterwarnings('ignore')


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         title: str = 'Confusion Matrix') -> plt.Figure:
    """
    Plot confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Risk', 'Risk'],
                yticklabels=['No Risk', 'Risk'], ax=ax)
    
    ax.set_title(title, fontweight='bold')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    
    return fig


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, 
                  title: str = 'ROC Curve') -> plt.Figure:
    """
    Plot ROC curve
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title, fontweight='bold')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_feature_importance(feature_names: List[str], 
                          importances: np.ndarray, 
                          top_n: int = 20) -> plt.Figure:
    """
    Plot feature importance
    """
    # Sort features by importance
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(range(len(top_features)), top_importances)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features)
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {top_n} Feature Importances', fontweight='bold')
    ax.invert_yaxis()
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, top_importances)):
        ax.text(value, i, f' {value:.4f}', va='center')
    
    plt.tight_layout()
    return fig


def calculate_business_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                             avg_loan_amount: float = 500000,
                             default_rate: float = 0.123,
                             loss_given_default: float = 0.6) -> Dict[str, Any]:
    """
    Calculate business impact metrics
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Basic metrics
    metrics = {
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),
        'total': int(tn + fp + fn + tp),
        'approval_rate': (tn + fn) / (tn + fp + fn + tp),
        'rejection_rate': (tp + fp) / (tn + fp + fn + tp)
    }
    
    # Business impact
    prevented_defaults = tp * default_rate
    reduced_losses = prevented_defaults * avg_loan_amount * loss_given_default
    false_positives_cost = fp * avg_loan_amount * 0.1  # 10% profit margin
    
    metrics.update({
        'expected_prevented_defaults': float(prevented_defaults),
        'potential_loss_reduction': float(reduced_losses),
        'lost_opportunity_cost': float(false_positives_cost),
        'net_benefit': float(reduced_losses - false_positives_cost)
    })
    
    return metrics


def save_results(results: Dict[str, Any], filename: str = None) -> str:
    """
    Save prediction results to JSON file
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'predictions_{timestamp}.json'
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    return filename


def load_results(filename: str) -> Dict[str, Any]:
    """
    Load prediction results from JSON file
    """
    with open(filename, 'r') as f:
        results = json.load(f)
    
    return results


def create_sample_data(n_samples: int = 100) -> pd.DataFrame:
    """
    Create sample data for testing
    """
    np.random.seed(42)
    
    data = {
        'Income': np.random.randint(500000, 10000000, n_samples),
        'Age': np.random.randint(20, 70, n_samples),
        'Experience': np.random.randint(0, 50, n_samples),
        'Married/Single': np.random.choice(['single', 'married'], n_samples),
        'House_Ownership': np.random.choice(['rented', 'owned', 'norent_noown'], n_samples),
        'Car_Ownership': np.random.choice(['yes', 'no'], n_samples),
        'Profession': np.random.choice(['Mechanical_engineer', 'Software_Developer', 
                                       'Technical_writer', 'Civil_servant'], n_samples),
        'CITY': np.random.choice(['Mumbai', 'Delhi', 'Bangalore', 'Chennai'], n_samples),
        'STATE': np.random.choice(['Maharashtra', 'Delhi', 'Karnataka', 'Tamil Nadu'], n_samples),
        'CURRENT_JOB_YRS': np.random.randint(0, 20, n_samples),
        'CURRENT_HOUSE_YRS': np.random.randint(0, 30, n_samples),
        'Risk_Flag': np.random.choice([0, 1], n_samples, p=[0.88, 0.12])
    }
    
    return pd.DataFrame(data)

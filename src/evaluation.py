"""
evaluation.py
Comprehensive model evaluation with all important metrics.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import os

def evaluate_model(model, X_test, y_test, model_name, label_map):
    """
    Full evaluation of a trained model.
    Prints and returns all metrics.
    """
    label_names = ['Low', 'Medium', 'High']
    
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # ROC-AUC (one-vs-rest for multiclass)
    try:
        roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')
    except:
        roc_auc = None
    
    print(f"\n{'='*55}")
    print(f"📈 EVALUATION: {model_name}")
    print(f"{'='*55}")
    print(f"  Accuracy:        {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  F1 Score (wtd):  {f1:.4f}")
    if roc_auc:
        print(f"  ROC-AUC (OvR):   {roc_auc:.4f}")
    
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_names))
    
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'y_pred': y_pred,
        'y_prob': y_prob
    }

def plot_confusion_matrix(y_test, y_pred, model_name, save=True):
    """Plot and optionally save confusion matrix heatmap."""
    label_names = ['Low', 'Medium', 'High']
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=label_names, yticklabels=label_names,
        linewidths=0.5
    )
    plt.title(f'Confusion Matrix — {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('Actual Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save:
        os.makedirs('images', exist_ok=True)
        path = f'images/confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  📸 Saved: {path}")
    
    plt.show()

def compare_models(results_list):
    """
    Print side-by-side comparison of all models.
    Returns name of best model by F1 score.
    """
    print("\n🏆 MODEL COMPARISON")
    print("="*55)
    print(f"{'Model':<25} {'Accuracy':>10} {'F1 Score':>10} {'ROC-AUC':>10}")
    print("-"*55)
    
    best_model = None
    best_f1 = 0
    
    for result in results_list:
        name = result['model_name']
        acc = result['accuracy']
        f1 = result['f1_score']
        roc = result['roc_auc'] if result['roc_auc'] else 0
        
        print(f"  {name:<23} {acc:>10.4f} {f1:>10.4f} {roc:>10.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_model = name
    
    print(f"\n✅ Best Model: {best_model} (F1 = {best_f1:.4f})")
    return best_model
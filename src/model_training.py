"""
model_training.py
Trains Random Forest and XGBoost classifiers.
Saves best model to disk.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib
import os

def split_data(df, feature_cols, target_col='performance_label_encoded', test_size=0.2):
    """
    Split data into train and test sets.
    Uses stratify to preserve class distribution in both sets.
    """
    X = df[feature_cols]
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"✅ Data split:")
    print(f"   Train: {X_train.shape[0]} samples")
    print(f"   Test:  {X_test.shape[0]} samples")
    print(f"\n   Class distribution (train):")
    for label, count in y_train.value_counts().sort_index().items():
        label_name = {0: 'Low', 1: 'Medium', 2: 'High'}[label]
        print(f"   {label_name}: {count} ({count/len(y_train)*100:.1f}%)")
    
    return X_train, X_test, y_train, y_test

def train_random_forest(X_train, y_train):
    """
    Train Random Forest Classifier.
    Random Forest = ensemble of decision trees with bagging.
    Good for: handles non-linearity, feature importance, robust.
    """
    print("\n🌲 Training Random Forest...")
    
    rf_model = RandomForestClassifier(
        n_estimators=200,          # 200 decision trees
        max_depth=10,              # Prevent overfitting
        min_samples_split=5,       # Minimum samples to split a node
        min_samples_leaf=2,        # Minimum samples in a leaf
        class_weight='balanced',   # Handle class imbalance
        random_state=42,
        n_jobs=-1                  # Use all CPU cores
    )
    
    rf_model.fit(X_train, y_train)
    
    # Cross-validation score
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=cv, scoring='f1_weighted')
    print(f"   CV F1 Score (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    return rf_model

def train_xgboost(X_train, y_train):
    """
    Train XGBoost Classifier.
    XGBoost = gradient boosting — sequentially fixes errors of previous trees.
    Good for: often highest accuracy, kaggle competitions, production systems.
    """
    print("\n⚡ Training XGBoost...")
    
    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='mlogloss',    # Multi-class log loss
        random_state=42,
        n_jobs=-1
    )
    
    xgb_model.fit(X_train, y_train, verbose=False)
    
    # Cross-validation score
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=cv, scoring='f1_weighted')
    print(f"   CV F1 Score (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    return xgb_model

def save_model(model, filename):
    """Save trained model to disk using joblib."""
    os.makedirs('models', exist_ok=True)
    filepath = f'models/{filename}'
    joblib.dump(model, filepath)
    print(f"   💾 Model saved: {filepath}")

def train_all_models(df, feature_cols):
    """
    Master training function.
    Trains both models and saves them.
    """
    print("\n🚀 STARTING MODEL TRAINING")
    print("="*50)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(df, feature_cols)
    
    # Train models
    rf_model = train_random_forest(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)
    
    # Save models
    print("\n💾 Saving Models:")
    save_model(rf_model, 'random_forest_model.pkl')
    save_model(xgb_model, 'xgboost_model.pkl')
    
    return rf_model, xgb_model, X_train, X_test, y_train, y_test, feature_cols
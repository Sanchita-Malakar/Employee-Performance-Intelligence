"""
preprocessing.py
Handles all data cleaning, encoding, and scaling tasks.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

def load_data(filepath):
    """Load dataset from CSV file."""
    df = pd.read_csv(filepath)
    print(f"✅ Data loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    return df

def check_data_quality(df):
    """Print data quality report."""
    print("\n📊 DATA QUALITY REPORT")
    print("="*50)
    print(f"Total rows: {df.shape[0]}")
    print(f"Total columns: {df.shape[1]}")
    print(f"\nMissing Values:")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        for col, count in missing.items():
            print(f"  {col}: {count} missing ({count/len(df)*100:.1f}%)")
    else:
        print("  No missing values found.")
    print(f"\nData Types:")
    print(df.dtypes)
    print(f"\nDuplicates: {df.duplicated().sum()}")

def handle_missing_values(df):
    """
    Fill missing values with appropriate strategies.
    - Numerical: fill with median (robust to outliers)
    - Categorical: fill with mode
    """
    df = df.copy()
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"  Filled '{col}' nulls with median: {median_val:.2f}")
    
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            print(f"  Filled '{col}' nulls with mode: {mode_val}")
    
    return df

def remove_outliers_iqr(df, columns):
    """
    Remove outliers using IQR method for specified columns.
    Keeps rows within [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
    """
    original_len = len(df)
    df = df.copy()
    
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    
    removed = original_len - len(df)
    print(f"  Outliers removed: {removed} rows ({removed/original_len*100:.1f}%)")
    return df

def encode_features(df):
    """
    Encode categorical variables:
    - Department: One-Hot Encoding (no ordinal relationship)
    - Performance label: Label Encoding (ordinal: Low=0, Medium=1, High=2)
    """
    df = df.copy()
    
    # One-hot encode department
    df = pd.get_dummies(df, columns=['department'], prefix='dept', drop_first=False)
    print(f"  Department one-hot encoded.")
    
    # Label encode target variable
    label_map = {'Low': 0, 'Medium': 1, 'High': 2}
    df['performance_label_encoded'] = df['performance_label'].map(label_map)
    print(f"  Performance label encoded: Low=0, Medium=1, High=2")
    
    return df, label_map

def scale_features(df, feature_cols, save_scaler=True):
    """
    Standardize numerical features using StandardScaler.
    Mean=0, Std=1 — important for distance-based models.
    """
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    if save_scaler:
        os.makedirs('models', exist_ok=True)
        joblib.dump(scaler, 'models/scaler.pkl')
        print(f"  Scaler saved to models/scaler.pkl")
    
    return df_scaled, scaler

def preprocess_pipeline(filepath):
    """
    Master preprocessing pipeline.
    Runs all steps in correct order and returns clean data.
    """
    print("\n🔧 STARTING PREPROCESSING PIPELINE")
    print("="*50)
    
    # Step 1: Load
    df = load_data(filepath)
    
    # Step 2: Quality check
    check_data_quality(df)
    
    # Step 3: Drop unnecessary columns
    df = df.drop(columns=['employee_id'])  # ID has no predictive value
    print("\n✅ Dropped employee_id column")
    
    # Step 4: Handle missing values
    print("\n📌 Handling Missing Values:")
    df = handle_missing_values(df)
    
    # Step 5: Remove outliers from salary
    print("\n📌 Removing Outliers:")
    df = remove_outliers_iqr(df, columns=['salary', 'avg_monthly_hours'])
    
    # Step 6: Encode features
    print("\n📌 Encoding Features:")
    df, label_map = encode_features(df)
    
    # Step 7: Define feature columns
    drop_cols = ['performance_label', 'performance_label_encoded', 'performance_score']
    feature_cols = [col for col in df.columns if col not in drop_cols]
    
    # Step 8: Scale features
    print("\n📌 Scaling Features:")
    df_scaled, scaler = scale_features(df, feature_cols)
    
    # Step 9: Save cleaned data
    os.makedirs('data/processed', exist_ok=True)
    df.to_csv('data/processed/cleaned_data.csv', index=False)
    print(f"\n✅ Cleaned data saved: {df.shape}")
    
    return df, df_scaled, feature_cols, label_map


if __name__ == "__main__":
    df, df_scaled, feature_cols, label_map = preprocess_pipeline('data/raw/employee_data.csv')
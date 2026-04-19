"""
main.py
====================================================
Employee Performance Predictor — Master Run Script
====================================================
Author: [Your Name]
Project: Employee Performance Prediction using ML
Dataset: Synthetic HR Data (1000 employees)
Models: Random Forest + XGBoost
====================================================
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

# Import project modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.data_generator import generate_employee_data
from src.preprocessing import preprocess_pipeline, handle_missing_values, encode_features
from src.feature_engineering import create_features, select_features
from src.model_training import split_data, train_random_forest, train_xgboost, save_model
from src.evaluation import evaluate_model, plot_confusion_matrix, compare_models
from src.visualizations import (
    plot_performance_distribution, plot_correlation_heatmap,
    plot_feature_importance, plot_performance_by_department,
    plot_salary_vs_performance, plot_shap_summary
)

def run_prediction_on_new_employee(model, scaler, feature_cols, label_map):
    """
    Simulate predicting performance for a NEW employee.
    This is the 'inference' or 'production' step.
    """
    print("\n" + "="*55)
    print("🎯 PREDICTING FOR NEW EMPLOYEE RECORDS")
    print("="*55)
    
    # Reverse label map
    reverse_label = {v: k for k, v in label_map.items()}
    
    # Simulate 5 new employees (manually crafted to test different outcomes)
    new_employees = pd.DataFrame({
        'age': [28, 45, 32, 55, 24],
        'experience_years': [5, 20, 8, 30, 2],
        'salary': [55000, 95000, 62000, 110000, 35000],
        'training_hours': [60, 10, 40, 5, 70],
        'satisfaction_score': [8.5, 3.2, 7.0, 4.5, 9.1],
        'attendance_pct': [97, 72, 88, 75, 99],
        'projects_completed': [12, 4, 9, 6, 15],
        'avg_monthly_hours': [200, 270, 220, 280, 180],
        'promotions_last_5yrs': [1, 0, 1, 0, 0],
        'work_accidents': [0, 1, 0, 1, 0],
        'left_company': [0, 1, 0, 0, 0],
        # Department dummies (one-hot) — assuming 6 departments
        'dept_Engineering': [1, 0, 0, 0, 0],
        'dept_Finance': [0, 0, 0, 1, 0],
        'dept_HR': [0, 0, 0, 0, 0],
        'dept_Marketing': [0, 0, 0, 0, 1],
        'dept_Operations': [0, 0, 1, 0, 0],
        'dept_Sales': [0, 1, 0, 0, 0],
    })
    
    # Add engineered features
    new_employees['salary_per_year_exp'] = new_employees['salary'] / (new_employees['experience_years'] + 1)
    new_employees['engagement_score'] = (
        (new_employees['satisfaction_score'] / 10 * 50) + 
        (new_employees['attendance_pct'] / 100 * 50)
    )
    new_employees['training_per_project'] = new_employees['training_hours'] / (new_employees['projects_completed'] + 1)
    new_employees['overwork_flag'] = (new_employees['avg_monthly_hours'] > 250).astype(int)
    new_employees['career_momentum'] = new_employees['promotions_last_5yrs'] / (new_employees['experience_years'] + 1)
    new_employees['early_starter'] = (new_employees['age'] - new_employees['experience_years'] < 24).astype(int)
    
    # Align columns with training features
    for col in feature_cols:
        if col not in new_employees.columns:
            new_employees[col] = 0
    
    new_employees_aligned = new_employees[feature_cols]
    
    # Scale
    new_employees_scaled = scaler.transform(new_employees_aligned)
    
    # Predict
    predictions = model.predict(new_employees_scaled)
    probabilities = model.predict_proba(new_employees_scaled)
    
    # Display results
    print(f"\n{'Emp':<6} {'Pred Label':<12} {'P(Low)':>8} {'P(Med)':>8} {'P(High)':>8} {'HR Action'}")
    print("-"*65)
    
    hr_actions = {
        'High': '⭐ Promotion candidate',
        'Medium': '📈 Assign stretch goals',
        'Low': '🔴 Immediate training plan'
    }
    
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        label = reverse_label[pred]
        action = hr_actions[label]
        print(f"  E{i+1:<4} {label:<12} {prob[0]:>8.3f} {prob[1]:>8.3f} {prob[2]:>8.3f}  {action}")
    
    # Save predictions
    os.makedirs('outputs', exist_ok=True)
    results = pd.DataFrame({
        'employee': [f'E{i+1}' for i in range(len(predictions))],
        'predicted_performance': [reverse_label[p] for p in predictions],
        'prob_low': probabilities[:, 0],
        'prob_medium': probabilities[:, 1],
        'prob_high': probabilities[:, 2]
    })
    results.to_csv('outputs/predictions.csv', index=False)
    print(f"\n💾 Predictions saved: outputs/predictions.csv")


def main():
    print("🚀 EMPLOYEE PERFORMANCE PREDICTOR")
    print("="*55)
    print("Starting full pipeline execution...\n")
    
    # ─── PHASE 1: Generate Data ───────────────────────────
    print("📦 PHASE 1: Generating Synthetic HR Dataset...")
    os.makedirs('data/raw', exist_ok=True)
    df_raw = generate_employee_data(n_employees=1000)
    df_raw.to_csv('data/raw/employee_data.csv', index=False)
    print(f"  Dataset: {df_raw.shape[0]} employees × {df_raw.shape[1]} features")
    
    # ─── PHASE 2: Preprocessing ───────────────────────────
    print("\n🔧 PHASE 2: Preprocessing...")
    df_clean, df_scaled, feature_cols_init, label_map = preprocess_pipeline('data/raw/employee_data.csv')
    
    # ─── PHASE 3: Visualize Raw Data ──────────────────────
    print("\n📊 PHASE 3: Generating EDA Visualizations...")
    plot_performance_distribution(df_raw)
    plot_correlation_heatmap(df_clean)
    plot_performance_by_department(df_raw)
    plot_salary_vs_performance(df_raw)
    
    # ─── PHASE 4: Feature Engineering ─────────────────────
    print("\n⚙️  PHASE 4: Feature Engineering...")
    df_clean = create_features(df_clean)
    top_features = select_features(df_clean, target_col='performance_label_encoded', top_k=18)
    
    # ─── PHASE 5: Model Training ───────────────────────────
    print("\n🤖 PHASE 5: Training Models...")
    X_train, X_test, y_train, y_test = split_data(df_clean, top_features)
    
    rf_model = train_random_forest(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)
    
    save_model(rf_model, 'random_forest_model.pkl')
    save_model(xgb_model, 'xgboost_model.pkl')
    
    # ─── PHASE 6: Evaluation ──────────────────────────────
    print("\n📈 PHASE 6: Model Evaluation...")
    rf_results = evaluate_model(rf_model, X_test, y_test, 'Random Forest', label_map)
    xgb_results = evaluate_model(xgb_model, X_test, y_test, 'XGBoost', label_map)
    
    plot_confusion_matrix(y_test, rf_results['y_pred'], 'Random Forest')
    plot_confusion_matrix(y_test, xgb_results['y_pred'], 'XGBoost')
    
    best_model_name = compare_models([rf_results, xgb_results])
    best_model = rf_model if 'Random' in best_model_name else xgb_model
    
    # ─── PHASE 7: Feature Importance + SHAP ───────────────
    print("\n🔍 PHASE 7: Feature Insights...")
    plot_feature_importance(rf_model, top_features, 'Random Forest')
    plot_shap_summary(xgb_model, X_test, top_features, 'XGBoost')
    
    # ─── PHASE 8: Predict New Employees ───────────────────
    scaler = joblib.load('models/scaler.pkl')
    run_prediction_on_new_employee(best_model, scaler, top_features, label_map)
    
    # ─── DONE ─────────────────────────────────────────────
    print("\n" + "="*55)
    print("✅ PROJECT COMPLETE!")
    print("="*55)
    print("\n📁 Output files generated:")
    print("  • data/raw/employee_data.csv")
    print("  • data/processed/cleaned_data.csv")
    print("  • models/random_forest_model.pkl")
    print("  • models/xgboost_model.pkl")
    print("  • images/ (6+ visualization charts)")
    print("  • outputs/predictions.csv")
    print("\n🎓 Ready to push to GitHub!")


if __name__ == "__main__":
    main()
"""
feature_engineering.py
Creates meaningful new features from existing ones.
Feature engineering often has more impact than algorithm choice.
"""

import pandas as pd
import numpy as np

def create_features(df):
    """
    Create derived features that capture meaningful business patterns.
    
    New features:
    - salary_per_year_exp: Efficiency of salary relative to experience
    - training_efficiency: Training hours relative to projects completed
    - work_life_balance: Monthly hours normalized
    - engagement_score: Composite of satisfaction + attendance
    - career_growth_rate: Promotions relative to experience
    """
    df = df.copy()
    
    # Salary efficiency (are they well-paid for their experience?)
    df['salary_per_year_exp'] = df['salary'] / (df['experience_years'] + 1)
    
    # Engagement composite score
    df['engagement_score'] = (
        (df['satisfaction_score'] / 10 * 50) + 
        (df['attendance_pct'] / 100 * 50)
    )
    
    # Training efficiency (training hours per project)
    df['training_per_project'] = df['training_hours'] / (df['projects_completed'] + 1)
    
    # Overwork indicator
    df['overwork_flag'] = (df['avg_monthly_hours'] > 250).astype(int)
    
    # Career momentum
    df['career_momentum'] = df['promotions_last_5yrs'] / (df['experience_years'] + 1)
    
    # Age-experience gap (how early did they start?)
    df['early_starter'] = (df['age'] - df['experience_years'] < 24).astype(int)
    
    print("✅ New features created:")
    new_cols = ['salary_per_year_exp', 'engagement_score', 'training_per_project',
                'overwork_flag', 'career_momentum', 'early_starter']
    for col in new_cols:
        print(f"   → {col}")
    
    return df

def select_features(df, target_col='performance_label_encoded', top_k=20):
    """
    Select top features using correlation with target.
    Returns list of most relevant feature column names.
    """
    drop_cols = ['performance_label', 'performance_label_encoded', 'performance_score']
    feature_cols = [col for col in df.columns if col not in drop_cols]
    
    # Compute correlation of each feature with target
    correlations = {}
    for col in feature_cols:
        if df[col].dtype in [np.float64, np.int64, float, int]:
            corr = abs(df[col].corr(df[target_col]))
            correlations[col] = corr
    
    # Sort by correlation
    sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n📊 Top {top_k} Features by Correlation with Performance:")
    for feat, corr in sorted_features[:top_k]:
        bar = '█' * int(corr * 30)
        print(f"  {feat:<35} {corr:.3f}  {bar}")
    
    top_features = [feat for feat, _ in sorted_features[:top_k]]
    return top_features


if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv('data/processed/cleaned_data.csv')
    df = create_features(df)
    top_features = select_features(df)
    print(f"\nSelected {len(top_features)} features for modeling.")
"""
data_generator.py
Generates a realistic synthetic HR dataset with 1000 employee records.
Uses controlled randomness to simulate real-world HR data distributions.
"""

import pandas as pd
import numpy as np

def generate_employee_data(n_employees=1000, random_state=42):
    """
    Generate synthetic employee performance dataset.
    
    Parameters:
        n_employees: Number of employee records to generate
        random_state: Seed for reproducibility
    
    Returns:
        DataFrame with employee features and performance labels
    """
    np.random.seed(random_state)
    
    # --- Department list ---
    departments = ['Engineering', 'Sales', 'Marketing', 'HR', 'Finance', 'Operations']
    
    # --- Generate base features ---
    age = np.random.randint(22, 60, n_employees)
    experience_years = np.clip(age - np.random.randint(20, 25, n_employees), 0, 35)
    department = np.random.choice(departments, n_employees)
    
    # Salary based on experience + department (realistic correlation)
    base_salary = 30000 + (experience_years * 2000)
    dept_multiplier = {
        'Engineering': 1.4, 'Finance': 1.2, 'Sales': 1.1,
        'Marketing': 1.0, 'HR': 0.9, 'Operations': 0.95
    }
    salary = np.array([
        base_salary[i] * dept_multiplier[department[i]] + np.random.randint(-3000, 3000)
        for i in range(n_employees)
    ])
    
    # Training hours (0-80 per year)
    training_hours = np.random.randint(0, 80, n_employees)
    
    # Satisfaction score (1-10)
    satisfaction_score = np.round(np.random.uniform(1, 10, n_employees), 1)
    
    # Attendance percentage (60-100%)
    attendance_pct = np.round(np.random.uniform(60, 100, n_employees), 1)
    
    # Number of projects completed
    projects_completed = np.random.randint(1, 20, n_employees)
    
    # Average monthly hours
    avg_monthly_hours = np.random.randint(140, 310, n_employees)
    
    # Number of promotions in last 5 years
    promotions = np.random.choice([0, 1, 2, 3], n_employees, p=[0.5, 0.3, 0.15, 0.05])
    
    # Work accidents (binary)
    work_accidents = np.random.choice([0, 1], n_employees, p=[0.85, 0.15])
    
    # Left company (for context, not used as target)
    left = np.random.choice([0, 1], n_employees, p=[0.76, 0.24])
    
    # --- Generate Performance Score (0-100) ---
    # This is the KEY step: performance is correlated with features
    performance_score = (
        0.25 * (training_hours / 80 * 100) +        # Training contribution
        0.20 * (satisfaction_score / 10 * 100) +    # Satisfaction contribution
        0.20 * (attendance_pct) +                    # Attendance contribution
        0.15 * (projects_completed / 20 * 100) +    # Projects contribution
        0.10 * np.clip(experience_years / 35 * 100, 0, 100) +  # Experience
        0.10 * np.random.uniform(0, 100, n_employees)  # Random noise
    )
    performance_score = np.clip(performance_score, 0, 100)
    
    # --- Convert performance score to category ---
    def score_to_label(score):
        if score >= 70:
            return 'High'
        elif score >= 45:
            return 'Medium'
        else:
            return 'Low'
    
    performance_label = [score_to_label(s) for s in performance_score]
    
    # --- Build DataFrame ---
    df = pd.DataFrame({
        'employee_id': [f'EMP{str(i).zfill(4)}' for i in range(1, n_employees + 1)],
        'age': age,
        'experience_years': experience_years,
        'department': department,
        'salary': salary.astype(int),
        'training_hours': training_hours,
        'satisfaction_score': satisfaction_score,
        'attendance_pct': attendance_pct,
        'projects_completed': projects_completed,
        'avg_monthly_hours': avg_monthly_hours,
        'promotions_last_5yrs': promotions,
        'work_accidents': work_accidents,
        'left_company': left,
        'performance_score': np.round(performance_score, 1),
        'performance_label': performance_label
    })
    
    # --- Inject 5% missing values for realism ---
    for col in ['satisfaction_score', 'training_hours', 'attendance_pct']:
        missing_idx = np.random.choice(df.index, size=int(0.05 * n_employees), replace=False)
        df.loc[missing_idx, col] = np.nan
    
    return df


if __name__ == "__main__":
    df = generate_employee_data()
    df.to_csv('data/raw/employee_data.csv', index=False)
    print(f"✅ Dataset generated: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nPerformance Label Distribution:")
    print(df['performance_label'].value_counts())
    print(f"\nFirst 3 rows:")
    print(df.head(3))
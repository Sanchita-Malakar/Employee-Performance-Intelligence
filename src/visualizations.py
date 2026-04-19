"""
visualizations.py
Generates all analysis and insight visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import shap
import os

# Set style globally
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
os.makedirs('images', exist_ok=True)

def plot_performance_distribution(df):
    """Bar chart of performance label counts."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Count plot
    counts = df['performance_label'].value_counts()
    colors = {'High': '#2ecc71', 'Medium': '#f39c12', 'Low': '#e74c3c'}
    bar_colors = [colors[label] for label in counts.index]
    
    axes[0].bar(counts.index, counts.values, color=bar_colors, edgecolor='black', linewidth=0.5)
    axes[0].set_title('Employee Performance Distribution', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Performance Category')
    axes[0].set_ylabel('Number of Employees')
    for i, (label, val) in enumerate(counts.items()):
        axes[0].text(i, val + 5, str(val), ha='center', fontweight='bold')
    
    # Pie chart
    axes[1].pie(counts.values, labels=counts.index, colors=bar_colors,
               autopct='%1.1f%%', startangle=90,
               wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    axes[1].set_title('Performance Share (%)', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('images/performance_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("📸 Saved: images/performance_distribution.png")

def plot_correlation_heatmap(df):
    """Correlation heatmap for numerical features."""
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Drop encoded label and score for cleaner view
    cols_to_drop = ['performance_label_encoded', 'performance_score', 'left_company', 'work_accidents']
    numeric_df = numeric_df.drop(columns=[c for c in cols_to_drop if c in numeric_df.columns])
    
    plt.figure(figsize=(12, 9))
    mask = np.triu(np.ones_like(numeric_df.corr(), dtype=bool))
    
    sns.heatmap(
        numeric_df.corr(), mask=mask, annot=True, fmt='.2f',
        cmap='RdYlGn', center=0, linewidths=0.5,
        annot_kws={'size': 9}, vmin=-1, vmax=1
    )
    plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('images/correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("📸 Saved: images/correlation_heatmap.png")

def plot_feature_importance(model, feature_cols, model_name='Random Forest'):
    """Horizontal bar chart of top 15 feature importances."""
    importances = model.feature_importances_
    feat_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importances
    }).sort_values('Importance', ascending=True).tail(15)
    
    plt.figure(figsize=(10, 7))
    bars = plt.barh(feat_df['Feature'], feat_df['Importance'], 
                    color=plt.cm.RdYlGn(feat_df['Importance'] / feat_df['Importance'].max()),
                    edgecolor='black', linewidth=0.4)
    
    plt.title(f'Top 15 Feature Importances — {model_name}', fontsize=13, fontweight='bold')
    plt.xlabel('Importance Score', fontsize=11)
    plt.tight_layout()
    plt.savefig('images/feature_importance.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("📸 Saved: images/feature_importance.png")

def plot_performance_by_department(df):
    """Stacked bar chart: performance levels by department."""
    if 'department' not in df.columns:
        print("  Note: Department column already encoded. Skipping dept plot.")
        return
    
    dept_perf = df.groupby(['department', 'performance_label']).size().unstack(fill_value=0)
    
    colors = {'High': '#2ecc71', 'Medium': '#f39c12', 'Low': '#e74c3c'}
    dept_perf.plot(kind='bar', stacked=True, figsize=(12, 6),
                   color=[colors.get(c, 'gray') for c in dept_perf.columns],
                   edgecolor='black', linewidth=0.5)
    
    plt.title('Performance Distribution by Department', fontsize=13, fontweight='bold')
    plt.xlabel('Department')
    plt.ylabel('Number of Employees')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Performance Level', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.savefig('images/performance_by_department.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("📸 Saved: images/performance_by_department.png")

def plot_shap_summary(model, X_test, feature_cols, model_name='XGBoost'):
    """
    SHAP (SHapley Additive exPlanations) summary plot.
    This is the most impressive chart for interviews — shows WHY model predicts each class.
    """
    print("\n🔍 Generating SHAP explanations (may take 30-60 seconds)...")
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test[:200])  # Use 200 samples for speed
    
    plt.figure(figsize=(10, 7))
    
    if isinstance(shap_values, list):
        # Multi-class: show for "High" performance (class index 2)
        shap.summary_plot(shap_values[2], X_test[:200], 
                         feature_names=feature_cols,
                         show=False, max_display=15,
                         plot_type="dot")
        plt.title(f'SHAP Values — {model_name} (Predicting High Performance)',
                 fontsize=12, fontweight='bold')
    else:
        shap.summary_plot(shap_values, X_test[:200],
                         feature_names=feature_cols,
                         show=False, max_display=15)
    
    plt.tight_layout()
    plt.savefig('images/shap_summary.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("📸 Saved: images/shap_summary.png")

def plot_salary_vs_performance(df):
    """Box plot of salary across performance levels."""
    if 'performance_label' not in df.columns:
        return
    
    plt.figure(figsize=(9, 6))
    order = ['Low', 'Medium', 'High']
    colors = ['#e74c3c', '#f39c12', '#2ecc71']
    
    sns.boxplot(data=df, x='performance_label', y='salary',
               order=order, palette=colors, linewidth=1.2)
    sns.stripplot(data=df, x='performance_label', y='salary',
                 order=order, color='black', alpha=0.15, size=2, jitter=True)
    
    plt.title('Salary Distribution by Performance Level', fontsize=13, fontweight='bold')
    plt.xlabel('Performance Label')
    plt.ylabel('Salary (₹ / $)')
    plt.tight_layout()
    plt.savefig('images/salary_vs_performance.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("📸 Saved: images/salary_vs_performance.png")
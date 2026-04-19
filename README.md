# Employee Performance Predictor

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**ML-powered HR analytics dashboard** that predicts employee performance (High / Medium / Low) using **Random Forest** and **XGBoost**. Built with synthetic HR data, feature engineering, and interactive visualizations – perfect for HR teams to identify high‑potential employees and at‑risk talent.

---

## 🚀 Features

- **Synthetic HR dataset** – Realistic employee records (age, department, salary, training hours, satisfaction, attendance, projects, etc.) with controlled correlations.
- **Two ML models** – Random Forest and XGBoost trained on the same data for comparison.
- **Interactive Streamlit dashboard** – Six pages for data exploration, EDA, model evaluation, real‑time prediction, and HR insights.
- **Feature engineering** – Engagement score, career momentum, salary efficiency, overwork flag, and more.
- **Model explainability** – SHAP summary plots and feature importance charts.
- **HR‑ready insights** – Promotion candidates, intervention lists, and department‑wise recommendations.

---

## 🛠️ Tech Stack

| Category       | Tools |
|----------------|-------|
| Frontend       | Streamlit, Matplotlib, Seaborn |
| ML Models      | Scikit‑learn (Random Forest), XGBoost |
| Explainability | SHAP |
| Data Handling  | Pandas, NumPy |
| Visualization  | Matplotlib, Seaborn |

---

## 📦 Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/Employee-Performance-Predictor.git
cd Employee-Performance-Predictor
2. Create a virtual environment (recommended)
bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
3. Install dependencies
bash
pip install -r requirements.txt
4. Run the dashboard
bash
streamlit run dashboard.py
The app will open in your default browser at http://localhost:8501.

Note: If you see an AttributeError: 'Styler' object has no attribute 'applymap', update the code in dashboard.py (line ~706) – replace .applymap with .map. This is a pandas version compatibility fix.

🧠 How It Works
Data Generation – generate_data() creates 1000 synthetic employee records. A performance score (0‑100) is calculated as a weighted sum of features + random noise, then discretised into Low (<45), Medium (45‑69), High (≥70).

Preprocessing & Feature Engineering

One‑hot encode department.

Create engineered features: engagement_score, salary_per_exp, training_per_project, overwork_flag, career_momentum.

Scale features with StandardScaler.

Model Training – Two classifiers are trained on the same train/test split (80/20, stratified):

Random Forest (200 trees, max_depth=10, class_weight='balanced')

XGBoost (200 estimators, max_depth=6, learning_rate=0.1)

Dashboard Pages

Overview – KPIs, model performance, workflow.

Dataset – Filter, view, and download the synthetic data.

EDA & Charts – Distributions, correlations, department analysis, salary insights.

Model Training – Metrics, confusion matrices, feature importance.

Predict Employee – Fill in employee details and get a real‑time prediction with probability scores and HR action.

HR Insights – Batch predictions on all employees, action lists, and strategic recommendations.

📊 Example Predictions
Employee	Department	Training Hours	Satisfaction	Predicted Label	HR Action
EMP0123	Engineering	60	8.5	High	⭐ Promotion candidate
EMP0456	Sales	15	3.2	Low	🔴 Immediate training plan
EMP0789	Marketing	40	6.5	Medium	📈 Assign stretch goals
📁 Project Structure
text
Employee-Performance-Predictor/
├── dashboard.py                 # Main Streamlit application (all‑in‑one)
├── main.py                      # Alternative script to run full pipeline + training
├── src/                         # Modular source files (optional, from main.py)
│   ├── data_generator.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── evaluation.py
│   └── visualizations.py
├── requirements.txt             # Python dependencies
├── .gitignore                   # Ignored files (models, data, cache)
├── README.md                    # This file
└── LICENSE                      # MIT license (optional)
🧪 Running the Full Pipeline (without dashboard)
If you prefer a command‑line version that trains models and prints evaluation metrics:

bash
python main.py
This will:

Generate data

Preprocess, engineer features

Train both models

Show classification reports, confusion matrices, SHAP plots

Save models and predictions

🐛 Known Issues & Fixes
Issue	Solution
AttributeError: 'Styler' object has no attribute 'applymap'	In dashboard.py, replace .applymap( with .map( (pandas ≥2.1.0).
Missing shap module	Run pip install shap (included in requirements.txt).
Slow first load	Caching is used; second run will be faster.
📈 Future Improvements (Roadmap)
Real data upload (CSV/Excel) with automatic column mapping

Hyperparameter tuning (GridSearchCV / Optuna)

Attrition risk prediction model

Docker deployment + Streamlit Cloud hosting

Individual prediction explanation with LIME

User authentication (HR / Manager roles)

🤝 Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

📜 License
This project is licensed under the MIT License – see the LICENSE file for details.

👤 Author
Sanchita Malakar



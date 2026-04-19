"""
dashboard.py  —  Employee Performance Predictor
Premium Redesign: Luxury Editorial Aesthetic
Run: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score,
                             confusion_matrix, classification_report,
                             roc_auc_score)
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════
#  PAGE CONFIG
# ════════════════════════════════════════════════
st.set_page_config(
    page_title="EmpIQ · Performance Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════
#  GLOBAL CSS
# ════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@300;400;500;600&family=Space+Mono:wght@400;700&display=swap');

:root {
  --ink:      #05070f;
  --ink-1:    #0b0e1a;
  --ink-2:    #111525;
  --ink-3:    #181d2e;
  --ink-4:    #1f253a;
  --border:   rgba(255,255,255,0.07);
  --gold:     #c9a84c;
  --gold-lt:  #e8c97a;
  --cyan:     #4fc4cf;
  --rose:     #e05c7a;
  --jade:     #4ecb8d;
  --amber:    #f5a623;
  --violet:   #a78bfa;
  --t0:       #f0f2fa;
  --t1:       #c2c8de;
  --t2:       #7b849e;
  --t3:       #454d62;
}

*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
  background: var(--ink) !important;
  font-family: 'Inter', sans-serif;
  color: var(--t0);
  -webkit-font-smoothing: antialiased;
}

#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"] { display:none !important; }

.block-container {
  padding: 1.5rem 2rem 3rem !important;
  max-width: 1440px !important;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
  background: var(--ink-1) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .stRadio { display:none !important; }
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div { color: var(--t1) !important; }
[data-testid="stSidebar"] .stSlider > div > div > div > div {
  background: var(--gold) !important;
}
/* Make nav buttons invisible but clickable */
[data-testid="stSidebar"] .stButton > button {
  position: absolute !important;
  opacity: 0 !important;
  width: 100% !important;
  height: 58px !important;
  top: -58px !important;
  left: 0 !important;
  cursor: pointer !important;
  z-index: 10 !important;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
  background: var(--ink-2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  padding: 4px !important;
  gap: 2px !important;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  color: var(--t2) !important;
  border-radius: 7px !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 0.82rem !important;
  font-weight: 500 !important;
  padding: 6px 18px !important;
  border: none !important;
}
.stTabs [aria-selected="true"] {
  background: rgba(201,168,76,0.12) !important;
  color: var(--gold-lt) !important;
  border: 1px solid rgba(201,168,76,0.28) !important;
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] { display:none !important; }

/* ── BUTTONS ── */
.stButton > button {
  background: linear-gradient(135deg,#c9a84c,#a8732e) !important;
  color: #05070f !important;
  border: none !important;
  border-radius: 10px !important;
  font-family: 'Space Grotesk', sans-serif !important;
  font-weight: 700 !important;
  font-size: 0.82rem !important;
  letter-spacing: 0.06em !important;
  text-transform: uppercase !important;
  padding: 0.65rem 2rem !important;
  box-shadow: 0 0 20px rgba(201,168,76,0.25) !important;
  transition: all 0.2s !important;
}
.stButton > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 6px 30px rgba(201,168,76,0.35) !important;
}

[data-testid="stDownloadButton"] > button {
  background: transparent !important;
  color: var(--gold) !important;
  border: 1px solid rgba(201,168,76,0.35) !important;
  border-radius: 10px !important;
  font-family: 'Space Grotesk', sans-serif !important;
  font-weight: 600 !important;
  font-size: 0.8rem !important;
  padding: 0.55rem 1.4rem !important;
  box-shadow: none !important;
}

/* ── INPUTS ── */
.stSelectbox > div > div,
.stMultiSelect > div > div {
  background: var(--ink-3) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  color: var(--t0) !important;
}
.stSlider > div > div > div > div { background: var(--gold) !important; }

/* ── DATAFRAME ── */
[data-testid="stDataFrame"] {
  border: 1px solid var(--border) !important;
  border-radius: 14px !important;
  overflow: hidden !important;
}

/* ── METRIC ── */
[data-testid="stMetric"] {
  background: var(--ink-2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 14px !important;
  padding: 1rem !important;
}

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--ink-1); }
::-webkit-scrollbar-thumb { background: var(--ink-4); border-radius: 10px; }

/* ── RESPONSIVE ── */
@media (max-width: 768px) {
  .block-container { padding: 0.8rem !important; }
}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════
#  PLOTLY THEME
# ════════════════════════════════════════════════
PL = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#c2c8de", size=12),
    colorway=["#c9a84c","#4fc4cf","#4ecb8d","#e05c7a","#f5a623","#a78bfa"],
    margin=dict(l=10, r=10, t=45, b=10),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.07)",
               tickfont=dict(size=11,color="#7b849e"), title_font=dict(color="#7b849e")),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.07)",
               tickfont=dict(size=11,color="#7b849e"), title_font=dict(color="#7b849e")),
    legend=dict(bgcolor="rgba(17,21,37,0.85)", bordercolor="rgba(255,255,255,0.07)",
                borderwidth=1, font=dict(color="#c2c8de",size=11)),
    hoverlabel=dict(bgcolor="#1f253a", bordercolor="#c9a84c",
                    font=dict(family="Space Mono",size=12,color="#f0f2fa")),
    title_font=dict(size=14, color="#f0f2fa", family="Space Grotesk"),
)
PC = {"High":"#4ecb8d","Medium":"#f5a623","Low":"#e05c7a"}

def pl(**kwargs):
    """Merge extra kwargs into plotly layout."""
    return {**PL, **kwargs}


# ════════════════════════════════════════════════
#  HTML HELPERS
# ════════════════════════════════════════════════
def kpi(label, value, sub, accent="#c9a84c"):
    return f"""
    <div style="background:linear-gradient(135deg,#111525,#181d2e);
      border:1px solid rgba(255,255,255,0.07);border-top:2px solid {accent};
      border-radius:16px;padding:1.4rem 1.6rem;position:relative;overflow:hidden;">
      <div style="position:absolute;top:-15px;right:8px;font-size:4.5rem;
        opacity:0.035;font-family:'Space Grotesk',sans-serif;font-weight:800;color:{accent};">◈</div>
      <div style="font-size:0.68rem;color:#7b849e;text-transform:uppercase;
        letter-spacing:0.1em;margin-bottom:0.45rem;font-family:'Space Mono',monospace;">{label}</div>
      <div style="font-family:'Space Grotesk',sans-serif;font-size:2.3rem;font-weight:800;
        color:{accent};line-height:1;margin-bottom:0.3rem;">{value}</div>
      <div style="font-size:0.72rem;color:#454d62;">{sub}</div>
    </div>"""

def sec(title, sub=""):
    s = f"<div style='font-size:0.75rem;color:#454d62;margin-top:3px;'>{sub}</div>" if sub else ""
    return f"""
    <div style="margin:2rem 0 1rem;display:flex;align-items:baseline;gap:0.9rem;">
      <div style="width:3px;height:1.3rem;background:linear-gradient(180deg,#c9a84c,transparent);
        border-radius:2px;flex-shrink:0;"></div>
      <div>
        <div style="font-family:'Space Grotesk',sans-serif;font-size:0.92rem;font-weight:700;
          color:#f0f2fa;text-transform:uppercase;letter-spacing:0.07em;">{title}</div>
        {s}
      </div>
    </div>"""

def badge(t, c="#4ecb8d"):
    return f"""<span style="display:inline-block;padding:0.22rem 0.8rem;border-radius:20px;
      font-family:'Space Mono',monospace;font-size:0.72rem;
      background:{c}18;color:{c};border:1px solid {c}40;">{t}</span>"""

def info(text):
    return f"""<div style="background:rgba(79,196,207,0.06);border:1px solid rgba(79,196,207,0.2);
      border-left:3px solid #4fc4cf;border-radius:10px;padding:0.85rem 1.1rem;
      color:#4fc4cf;font-size:0.82rem;margin:0.8rem 0;line-height:1.6;">{text}</div>"""

def page_title(title, color_word, subtitle):
    words = title.split()
    colored = title.replace(color_word, f'<span style="color:#c9a84c;">{color_word}</span>')
    return f"""
    <div style="padding:0 0 1.5rem;">
      <div style="font-family:'Space Grotesk',sans-serif;font-size:2rem;font-weight:800;
        color:#f0f2fa;line-height:1.2;">{colored}</div>
      <div style="font-size:0.82rem;color:#454d62;margin-top:0.4rem;">{subtitle}</div>
    </div>"""


# ════════════════════════════════════════════════
#  DATA & MODEL
# ════════════════════════════════════════════════
@st.cache_data
def generate_data(n=1000, seed=42):
    np.random.seed(seed)
    depts = ['Engineering','Sales','Marketing','HR','Finance','Operations']
    age  = np.random.randint(22,60,n)
    exp  = np.clip(age - np.random.randint(20,25,n), 0, 35)
    dept = np.random.choice(depts,n)
    base = 30000 + exp*2000
    mult = {'Engineering':1.4,'Finance':1.2,'Sales':1.1,
            'Marketing':1.0,'HR':0.9,'Operations':0.95}
    sal  = np.array([base[i]*mult[dept[i]]+np.random.randint(-3000,3000) for i in range(n)]).astype(int)
    trn  = np.random.randint(0,80,n)
    sat  = np.round(np.random.uniform(1,10,n),1)
    att  = np.round(np.random.uniform(60,100,n),1)
    proj = np.random.randint(1,20,n)
    moh  = np.random.randint(140,310,n)
    prom = np.random.choice([0,1,2,3],n,p=[0.5,0.3,0.15,0.05])
    acc  = np.random.choice([0,1],n,p=[0.85,0.15])
    sc   = np.clip(0.25*(trn/80*100)+0.20*(sat/10*100)+0.20*att
                   +0.15*(proj/20*100)+0.10*np.clip(exp/35*100,0,100)
                   +0.10*np.random.uniform(0,100,n),0,100)
    lbl  = ['High' if s>=70 else 'Medium' if s>=45 else 'Low' for s in sc]
    return pd.DataFrame({
        'employee_id'       :[f'EMP{str(i).zfill(4)}' for i in range(1,n+1)],
        'age':age,'experience_years':exp,'department':dept,'salary':sal,
        'training_hours':trn,'satisfaction_score':sat,'attendance_pct':att,
        'projects_completed':proj,'avg_monthly_hours':moh,
        'promotions_last_5yrs':prom,'work_accidents':acc,
        'performance_score':np.round(sc,1),'performance_label':lbl,
    })

@st.cache_resource
def train_models(df):
    d = df.copy()
    d = pd.get_dummies(d, columns=['department'], prefix='dept')
    lm = {'Low':0,'Medium':1,'High':2}
    d['target']          = d['performance_label'].map(lm)
    d['engagement']      = (d['satisfaction_score']/10*50)+(d['attendance_pct']/100*50)
    d['salary_per_exp']  = d['salary']/(d['experience_years']+1)
    d['train_per_proj']  = d['training_hours']/(d['projects_completed']+1)
    d['overwork']        = (d['avg_monthly_hours']>250).astype(int)
    d['career_momentum'] = d['promotions_last_5yrs']/(d['experience_years']+1)
    drop = ['employee_id','performance_label','performance_score','target']
    fc   = [c for c in d.columns if c not in drop]
    X    = d[fc].fillna(d[fc].median()); y = d['target']
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)
    rf  = RandomForestClassifier(n_estimators=200,max_depth=10,class_weight='balanced',
                                  random_state=42,n_jobs=-1)
    rf.fit(Xtr_s, ytr)
    xgb = XGBClassifier(n_estimators=200,max_depth=6,learning_rate=0.1,
                         use_label_encoder=False,eval_metric='mlogloss',
                         random_state=42,n_jobs=-1)
    xgb.fit(Xtr_s, ytr, verbose=False)
    return rf,xgb,sc,fc,Xtr_s,Xte_s,ytr,yte,lm


# ════════════════════════════════════════════════
#  SESSION STATE + SIDEBAR
# ════════════════════════════════════════════════
if "page" not in st.session_state:
    st.session_state.page = "Overview"

PAGES = [
    ("Overview",       "◈","Dashboard home"),
    ("Dataset",        "⊞","Browse & filter"),
    ("EDA & Charts",   "∿","Visual exploration"),
    ("Model Training", "⬡","Train & evaluate"),
    ("Predict",        "◎","Live prediction"),
    ("HR Insights",    "◆","Recommendations"),
]

with st.sidebar:
    st.markdown("""
    <div style="padding:2rem 1.5rem 1.4rem;border-bottom:1px solid rgba(255,255,255,0.06);">
      <div style="font-family:'Space Grotesk',sans-serif;font-size:1.6rem;font-weight:800;
        background:linear-gradient(135deg,#e8c97a,#c9a84c);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
        EmpIQ
      </div>
      <div style="font-size:0.65rem;color:#454d62;margin-top:2px;
        font-family:'Space Mono',monospace;letter-spacing:0.06em;">
        PERFORMANCE INTELLIGENCE
      </div>
    </div>
    <div style="padding:1rem 1.5rem 0.4rem;font-size:0.63rem;color:#454d62;
      text-transform:uppercase;letter-spacing:0.12em;font-family:'Space Mono',monospace;">
      Navigation
    </div>
    """, unsafe_allow_html=True)

    for pname, icon, desc in PAGES:
        active = st.session_state.page == pname
        bg  = "background:linear-gradient(90deg,rgba(201,168,76,0.1),transparent);border-left:2px solid #c9a84c;" if active else "background:transparent;border-left:2px solid transparent;"
        tc  = "#e8c97a" if active else "#7b849e"
        ic  = "#c9a84c" if active else "#3a4055"
        dc  = "#7b849e" if active else "#3a4055"
        st.markdown(f"""
        <div style="{bg}padding:0.72rem 1.5rem;display:flex;align-items:center;
          gap:0.75rem;transition:all 0.15s;cursor:pointer;">
          <span style="font-size:0.95rem;color:{ic};width:18px;text-align:center;">{icon}</span>
          <div>
            <div style="font-family:'Space Grotesk',sans-serif;font-size:0.78rem;font-weight:600;
              color:{tc};">{pname}</div>
            <div style="font-size:0.63rem;color:{dc};margin-top:1px;">{desc}</div>
          </div>
        </div>""", unsafe_allow_html=True)
        if st.button(f"·{pname}", key=f"nb_{pname}", use_container_width=True):
            st.session_state.page = pname
            st.rerun()

    st.markdown("""
    <div style="padding:1.2rem 1.5rem 0.4rem;border-top:1px solid rgba(255,255,255,0.05);
      margin-top:0.6rem;font-size:0.63rem;color:#454d62;text-transform:uppercase;
      letter-spacing:0.12em;font-family:'Space Mono',monospace;">Configuration</div>
    """, unsafe_allow_html=True)
    n_emp = st.slider("Dataset Size", 300, 2000, 1000, 100)

    st.markdown("""
    <div style="padding:1rem 1.5rem;border-top:1px solid rgba(255,255,255,0.05);margin-top:0.6rem;">
      <div style="font-size:0.63rem;color:#454d62;text-transform:uppercase;
        letter-spacing:0.12em;font-family:'Space Mono',monospace;margin-bottom:0.6rem;">Stack</div>
      <div style="display:flex;flex-wrap:wrap;gap:0.35rem;">
        <span style="font-size:0.66rem;padding:0.18rem 0.55rem;border-radius:5px;
          background:rgba(78,203,141,0.1);color:#4ecb8d;border:1px solid rgba(78,203,141,0.2);">Python</span>
        <span style="font-size:0.66rem;padding:0.18rem 0.55rem;border-radius:5px;
          background:rgba(79,196,207,0.1);color:#4fc4cf;border:1px solid rgba(79,196,207,0.2);">XGBoost</span>
        <span style="font-size:0.66rem;padding:0.18rem 0.55rem;border-radius:5px;
          background:rgba(201,168,76,0.1);color:#c9a84c;border:1px solid rgba(201,168,76,0.2);">RF</span>
        <span style="font-size:0.66rem;padding:0.18rem 0.55rem;border-radius:5px;
          background:rgba(224,92,122,0.1);color:#e05c7a;border:1px solid rgba(224,92,122,0.2);">Plotly</span>
        <span style="font-size:0.66rem;padding:0.18rem 0.55rem;border-radius:5px;
          background:rgba(167,139,250,0.1);color:#a78bfa;border:1px solid rgba(167,139,250,0.2);">SHAP</span>
      </div>
    </div>""", unsafe_allow_html=True)

page = st.session_state.page

# ════════════════════════════════════════════════
#  LOAD + TRAIN
# ════════════════════════════════════════════════
with st.spinner(""):
    df = generate_data(n_emp)
    rf,xgb,scaler,fcols,Xtr,Xte,ytr,yte,lmap = train_models(df)

rlmap = {v:k for k,v in lmap.items()}
rp = rf.predict(Xte);  xp = xgb.predict(Xte)
ra = accuracy_score(yte,rp); xa = accuracy_score(yte,xp)
rf1= f1_score(yte,rp,average='weighted'); xf1=f1_score(yte,xp,average='weighted')
try:
    rr = roc_auc_score(yte,rf.predict_proba(Xte),multi_class='ovr',average='weighted')
    xr = roc_auc_score(yte,xgb.predict_proba(Xte),multi_class='ovr',average='weighted')
except: rr=xr=0


# ════════════════════════════════════════════════════════════
#  ◈  OVERVIEW
# ════════════════════════════════════════════════════════════
if page == "Overview":
    hi=(df.performance_label=='High').sum()
    me=(df.performance_label=='Medium').sum()
    lo=(df.performance_label=='Low').sum()

    # Hero
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0b0e1a 0%,#111525 55%,#131928 100%);
      border:1px solid rgba(255,255,255,0.07);border-radius:20px;
      padding:3rem 3rem 2.5rem;margin-bottom:1.8rem;position:relative;overflow:hidden;">
      <div style="position:absolute;top:0;right:0;width:55%;height:100%;
        background:radial-gradient(ellipse at 85% 50%,rgba(201,168,76,0.07),transparent 65%);
        pointer-events:none;"></div>
      <div style="position:absolute;bottom:-60px;left:-40px;width:260px;height:260px;
        background:radial-gradient(circle,rgba(79,196,207,0.04),transparent 70%);
        pointer-events:none;"></div>
      <div style="position:relative;">
        <div style="font-size:0.68rem;color:#c9a84c;text-transform:uppercase;
          letter-spacing:0.16em;font-family:'Space Mono',monospace;margin-bottom:0.9rem;">
          ◈ &nbsp; HR ANALYTICS PLATFORM &nbsp; · &nbsp; v2.0
        </div>
        <div style="font-family:'Space Grotesk',sans-serif;font-size:3.2rem;font-weight:800;
          line-height:1.08;color:#f0f2fa;margin-bottom:0.6rem;">
          Employee<br>
          <span style="background:linear-gradient(90deg,#e8c97a,#c9a84c 55%,#a8732e);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            Performance
          </span> Intelligence
        </div>
        <p style="color:#7b849e;font-size:0.9rem;max-width:540px;line-height:1.75;margin-top:0.4rem;">
          ML-powered HR analytics built on a synthetic dataset of <b style="color:#c2c8de;">{n_emp:,} employees</b>.
          Predicts performance using Random Forest &amp; XGBoost with full explainability.
        </p>
        <div style="display:flex;gap:0.55rem;margin-top:1.4rem;flex-wrap:wrap;">
          {badge("Random Forest","#4ecb8d")}
          {badge("XGBoost","#4fc4cf")}
          {badge("Synthetic Data","#c9a84c")}
          {badge("Explainable AI","#a78bfa")}
          {badge("Plotly Charts","#e05c7a")}
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    # KPI row
    c1,c2,c3,c4 = st.columns(4)
    c1.markdown(kpi("Total Employees",f"{len(df):,}","Synthetic HR records","#4fc4cf"), unsafe_allow_html=True)
    c2.markdown(kpi("High Performers",str(hi),f"{hi/len(df)*100:.1f}% of workforce","#4ecb8d"), unsafe_allow_html=True)
    c3.markdown(kpi("Medium Performers",str(me),f"{me/len(df)*100:.1f}% of workforce","#f5a623"), unsafe_allow_html=True)
    c4.markdown(kpi("Low Performers",str(lo),f"{lo/len(df)*100:.1f}% of workforce","#e05c7a"), unsafe_allow_html=True)

    st.markdown(sec("Model Performance", "Live results from trained classifiers"), unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    for col,nm,acc,f1,roc,accent in [
        (c1,"Random Forest",ra,rf1,rr,"#4ecb8d"),
        (c2,"XGBoost",xa,xf1,xr,"#4fc4cf"),
    ]:
        with col:
            emo = "🌲" if "Forest" in nm else "⚡"
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#111525,#181d2e);
              border:1px solid rgba(255,255,255,0.07);border-radius:18px;padding:1.8rem 2rem;
              position:relative;overflow:hidden;">
              <div style="position:absolute;top:-20px;right:-10px;font-size:7rem;opacity:0.025;">{emo}</div>
              <div style="font-family:'Space Grotesk',sans-serif;font-size:0.88rem;font-weight:700;
                color:#f0f2fa;margin-bottom:1.3rem;">{emo}&nbsp; {nm}</div>
              <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:1rem;">
                {"".join([f'''<div style="text-align:center;background:rgba(0,0,0,0.2);
                  border-radius:10px;padding:0.8rem 0.4rem;">
                  <div style="font-size:0.6rem;color:#7b849e;text-transform:uppercase;
                    letter-spacing:0.1em;margin-bottom:0.25rem;font-family:Space Mono,monospace;">{lb}</div>
                  <div style="font-family:'Space Grotesk',sans-serif;font-size:1.7rem;
                    font-weight:800;color:{accent};">{vl}</div>
                </div>'''
                for lb,vl in [("Accuracy",f"{acc*100:.1f}%"),("F1 Score",f"{f1:.3f}"),("ROC-AUC",f"{roc:.3f}")]
                ])}
              </div>
              <div style="margin-top:1rem;height:3px;background:rgba(255,255,255,0.04);
                border-radius:2px;overflow:hidden;">
                <div style="height:100%;width:{acc*100:.0f}%;
                  background:linear-gradient(90deg,{accent},{accent}66);border-radius:2px;"></div>
              </div>
            </div>""", unsafe_allow_html=True)

    st.markdown(sec("Project Pipeline","End-to-end ML workflow"), unsafe_allow_html=True)
    steps=[
        ("01","Data Generation","Realistic synthetic HR dataset","#c9a84c"),
        ("02","Preprocessing","Encoding · Scaling · Outlier removal","#4fc4cf"),
        ("03","Feature Eng.","Engagement score · Career momentum","#4ecb8d"),
        ("04","Model Training","RF + XGBoost · Cross-validation","#f5a623"),
        ("05","Evaluation","Accuracy · F1 · CM · ROC-AUC","#e05c7a"),
        ("06","Explainability","Feature importance · SHAP values","#a78bfa"),
        ("07","Live Prediction","Real-time inference · HR action","#4fc4cf"),
        ("08","Portfolio Ready","Documented · GitHub-ready","#c9a84c"),
    ]
    cols = st.columns(4)
    for i,(num,title,desc,color) in enumerate(steps):
        with cols[i%4]:
            st.markdown(f"""
            <div style="background:#111525;border:1px solid rgba(255,255,255,0.06);
              border-top:2px solid {color};border-radius:14px;padding:1.1rem;margin-bottom:0.8rem;">
              <div style="font-family:'Space Mono',monospace;font-size:0.68rem;
                color:{color};margin-bottom:0.35rem;">{num}</div>
              <div style="font-family:'Space Grotesk',sans-serif;font-size:0.8rem;font-weight:700;
                color:#f0f2fa;margin-bottom:0.2rem;">{title}</div>
              <div style="font-size:0.7rem;color:#454d62;line-height:1.5;">{desc}</div>
            </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  ⊞  DATASET
# ════════════════════════════════════════════════════════════
elif page == "Dataset":
    st.markdown(page_title("Dataset Explorer","Explorer",
        f"{len(df):,} synthetic employee records · {df.shape[1]} features"), unsafe_allow_html=True)

    c1,c2,c3 = st.columns([2,2,3])
    with c1: dept_f = st.multiselect("Department",sorted(df.department.unique()),default=sorted(df.department.unique()))
    with c2: perf_f = st.multiselect("Performance",['High','Medium','Low'],default=['High','Medium','Low'])
    with c3: sal_r  = st.slider("Salary ($)",int(df.salary.min()),int(df.salary.max()),(int(df.salary.min()),int(df.salary.max())))

    fdf = df[df.department.isin(dept_f)&df.performance_label.isin(perf_f)&df.salary.between(*sal_r)]
    st.markdown(info(f"Showing <b>{len(fdf):,}</b> of <b>{len(df):,}</b> records"), unsafe_allow_html=True)

    def cp(v): return f"color:{'#4ecb8d' if v=='High' else '#f5a623' if v=='Medium' else '#e05c7a'};font-weight:600" if v in ('High','Medium','Low') else ""
    st.dataframe(fdf.style.map(cp,subset=['performance_label']),use_container_width=True,height=420)
    st.download_button("⬇ Export CSV",fdf.to_csv(index=False),"employees.csv","text/csv")

    st.markdown(sec("Statistical Summary"), unsafe_allow_html=True)
    st.dataframe(df.describe().round(2),use_container_width=True)


# ════════════════════════════════════════════════════════════
#  ∿  EDA & CHARTS
# ════════════════════════════════════════════════════════════
elif page == "EDA & Charts":
    st.markdown(page_title("Exploratory Data Analysis","Data",
        "Interactive Plotly charts — hover, zoom, pan, and export"), unsafe_allow_html=True)

    t1,t2,t3,t4 = st.tabs(["Distribution","Correlation","By Department","Salary & Experience"])

    with t1:
        c1,c2 = st.columns(2)
        with c1:
            cnt = df.performance_label.value_counts().reset_index()
            cnt.columns=['label','count']
            fig=px.bar(cnt,x='label',y='count',color='label',
                       color_discrete_map=PC,text='count',
                       title="Performance Distribution")
            fig.update_traces(textposition='outside',textfont_color='#c2c8de',
                              marker_line_width=0,width=0.5)
            fig.update_layout(**pl(showlegend=False))
            st.plotly_chart(fig,use_container_width=True)
        with c2:
            fig2=px.pie(cnt,values='count',names='label',color='label',
                        color_discrete_map=PC,hole=0.58,title="Workforce Composition")
            fig2.update_traces(textfont_color='#f0f2fa',textfont_size=12,
                               pull=[0.05 if l=='High' else 0 for l in cnt.label])
            fig2.update_layout(**pl())
            st.plotly_chart(fig2,use_container_width=True)
        c3,c4 = st.columns(2)
        with c3:
            fig3=px.histogram(df,x='performance_score',nbins=35,
                              color_discrete_sequence=['#c9a84c'],
                              title="Performance Score Distribution",opacity=0.85)
            fig3.add_vline(x=70,line_dash="dash",line_color="#4ecb8d",
                           annotation_text="High ≥70",annotation_font_color="#4ecb8d")
            fig3.add_vline(x=45,line_dash="dash",line_color="#e05c7a",
                           annotation_text="Low <45",annotation_font_color="#e05c7a")
            fig3.update_traces(marker_line_width=0)
            fig3.update_layout(**pl())
            st.plotly_chart(fig3,use_container_width=True)
        with c4:
            fig4=px.histogram(df,x='satisfaction_score',color='performance_label',
                              nbins=20,barmode='overlay',color_discrete_map=PC,
                              opacity=0.72,title="Satisfaction by Performance")
            fig4.update_traces(marker_line_width=0)
            fig4.update_layout(**pl())
            st.plotly_chart(fig4,use_container_width=True)

    with t2:
        num_df=df.select_dtypes(include=np.number).drop(columns=['performance_score'],errors='ignore')
        corr=num_df.corr().round(2)
        fig=px.imshow(corr,text_auto=True,aspect='auto',
                      color_continuous_scale=[[0,'#e05c7a'],[0.5,'#111525'],[1,'#4ecb8d']],
                      title="Feature Correlation Matrix",zmin=-1,zmax=1)
        fig.update_traces(textfont_size=9,textfont_color='#f0f2fa')
        fig.update_layout(**{k:v for k,v in pl().items() if k not in['xaxis','yaxis']},
                          height=520)
        fig.update_coloraxes(colorbar=dict(tickfont=dict(color='#7b849e')))
        st.plotly_chart(fig,use_container_width=True)
        cp2=num_df.corrwith(df.performance_score).abs().sort_values(ascending=True)
        fig2=go.Figure(go.Bar(x=cp2.values,y=cp2.index,orientation='h',
            marker=dict(color=cp2.values,colorscale=[[0,'#454d62'],[0.5,'#c9a84c'],[1,'#4ecb8d']],line_width=0),
            text=[f"{v:.3f}" for v in cp2.values],textposition='outside',
            textfont=dict(color='#c2c8de',size=10)))
        fig2.update_layout(**pl(),height=380,title="Correlations with Performance Score")
        st.plotly_chart(fig2,use_container_width=True)

    with t3:
        c1,c2=st.columns(2)
        with c1:
            dp=df.groupby(['department','performance_label']).size().reset_index(name='count')
            fig=px.bar(dp,x='department',y='count',color='performance_label',barmode='stack',
                       color_discrete_map=PC,title="Performance by Department")
            fig.update_traces(marker_line_width=0)
            fig.update_layout(**pl(),xaxis_tickangle=-30)
            st.plotly_chart(fig,use_container_width=True)
        with c2:
            da=df.groupby('department')['performance_score'].mean().reset_index()
            da.columns=['department','avg']
            da=da.sort_values('avg',ascending=True)
            fig2=go.Figure(go.Bar(x=da.avg,y=da.department,orientation='h',
                marker=dict(color=da.avg,colorscale=[[0,'#e05c7a'],[0.5,'#f5a623'],[1,'#4ecb8d']],line_width=0),
                text=[f"{v:.1f}" for v in da.avg],textposition='outside',
                textfont=dict(color='#c2c8de')))
            fig2.update_layout(**pl(),title="Avg Score by Department")
            st.plotly_chart(fig2,use_container_width=True)
        dt=df.groupby('department')['training_hours'].mean().reset_index()
        dt.columns=['department','avg_tr']
        fig3=px.bar(dt.sort_values('avg_tr',ascending=False),x='department',y='avg_tr',
                    color='avg_tr',color_continuous_scale=[[0,'#111525'],[1,'#4fc4cf']],
                    title="Avg Training Hours by Department")
        fig3.update_traces(marker_line_width=0)
        fig3.update_layout(**pl(),showlegend=False)
        st.plotly_chart(fig3,use_container_width=True)

    with t4:
        c1,c2=st.columns(2)
        with c1:
            fig=px.box(df,x='performance_label',y='salary',color='performance_label',
                       color_discrete_map=PC,points='outliers',
                       category_orders={'performance_label':['Low','Medium','High']},
                       title="Salary by Performance Level")
            fig.update_traces(marker_size=3,opacity=0.85,line_width=1.5)
            fig.update_layout(**pl(),showlegend=False)
            st.plotly_chart(fig,use_container_width=True)
        with c2:
            samp=df.sample(min(500,len(df)),random_state=42)
            fig2=px.scatter(samp,x='experience_years',y='salary',color='performance_label',
                            color_discrete_map=PC,size='performance_score',size_max=14,
                            opacity=0.72,title="Experience vs Salary",
                            hover_data=['department','training_hours'])
            fig2.update_traces(marker_line_width=0)
            fig2.update_layout(**pl())
            st.plotly_chart(fig2,use_container_width=True)
        fig3=px.violin(df,x='performance_label',y='training_hours',
                       color='performance_label',color_discrete_map=PC,
                       box=True,points=False,
                       category_orders={'performance_label':['Low','Medium','High']},
                       title="Training Hours Distribution by Performance")
        fig3.update_layout(**pl(),showlegend=False)
        st.plotly_chart(fig3,use_container_width=True)


# ════════════════════════════════════════════════════════════
#  ⬡  MODEL TRAINING
# ════════════════════════════════════════════════════════════
elif page == "Model Training":
    st.markdown(page_title("Model Training & Evaluation","Evaluation",
        "Random Forest vs XGBoost · Multi-class classification · Plotly visualizations"), unsafe_allow_html=True)

    t1,t2,t3 = st.tabs(["Metrics & Reports","Confusion Matrix","Feature Importance"])

    with t1:
        c1,c2=st.columns(2)
        for col,nm,acc,f1,roc,preds,ac in [
            (c1,"Random Forest",ra,rf1,rr,rp,"#4ecb8d"),
            (c2,"XGBoost",xa,xf1,xr,xp,"#4fc4cf")
        ]:
            with col:
                emo="🌲" if "Forest" in nm else "⚡"
                st.markdown(f"""
                <div style="background:linear-gradient(135deg,#111525,#181d2e);
                  border:1px solid rgba(255,255,255,0.07);border-radius:16px;
                  padding:1.5rem 1.8rem;margin-bottom:1rem;">
                  <div style="font-family:'Space Grotesk',sans-serif;font-size:0.88rem;font-weight:700;
                    color:#f0f2fa;margin-bottom:1.1rem;">{emo}&nbsp; {nm}</div>
                  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:0.8rem;">
                    {"".join([f'''<div style="text-align:center;background:rgba(0,0,0,0.2);
                      border-radius:10px;padding:0.7rem 0.4rem;">
                      <div style="font-size:0.6rem;color:#7b849e;text-transform:uppercase;
                        letter-spacing:0.08em;margin-bottom:0.2rem;font-family:Space Mono,monospace;">{lb}</div>
                      <div style="font-family:'Space Grotesk',sans-serif;font-size:1.5rem;
                        font-weight:800;color:{ac};">{vl}</div>
                    </div>'''
                    for lb,vl in [("Accuracy",f"{acc*100:.1f}%"),("F1",f"{f1:.3f}"),("ROC",f"{roc:.3f}")]
                    ])}
                  </div>
                </div>""", unsafe_allow_html=True)
                rep=classification_report(yte,preds,target_names=['Low','Medium','High'],output_dict=True)
                st.dataframe(pd.DataFrame(rep).T.round(3),use_container_width=True,height=190)

        st.markdown(sec("Model Comparison"), unsafe_allow_html=True)
        fig=go.Figure()
        cats=['Accuracy','F1 Score','ROC-AUC']
        for nm,vals,color in [("Random Forest",[ra,rf1,rr],"#4ecb8d"),("XGBoost",[xa,xf1,xr],"#4fc4cf")]:
            fig.add_trace(go.Scatterpolar(r=vals+[vals[0]],theta=cats+[cats[0]],fill='toself',
                name=nm,line=dict(color=color,width=2.5),fillcolor=f"rgba({'78,203,141' if 'Forest' in nm else '79,196,207'},0.08)"))
        fig.update_layout(**{k:v for k,v in pl().items() if k not in['xaxis','yaxis']},
            polar=dict(bgcolor='rgba(0,0,0,0)',
                radialaxis=dict(visible=True,range=[0.7,1],
                    gridcolor='rgba(255,255,255,0.07)',tickfont=dict(color='#7b849e',size=9)),
                angularaxis=dict(gridcolor='rgba(255,255,255,0.07)',
                    tickfont=dict(color='#c2c8de',size=11))),
            title="Performance Radar",height=380)
        st.plotly_chart(fig,use_container_width=True)

    with t2:
        c1,c2=st.columns(2)
        for col,preds,nm in [(c1,rp,"Random Forest"),(c2,xp,"XGBoost")]:
            cm=confusion_matrix(yte,preds)
            labs=['Low','Medium','High']
            fig=px.imshow(cm,x=labs,y=labs,text_auto=True,
                color_continuous_scale=[[0,'#111525'],[0.5,'rgba(201,168,76,0.5)'],[1,'#c9a84c']],
                title=f"Confusion Matrix — {nm}",aspect='equal')
            fig.update_traces(textfont_size=15,textfont_color='#f0f2fa',
                              textfont_family='Space Mono')
            fig.update_layout(**{k:v for k,v in pl().items() if k not in['xaxis','yaxis']},
                xaxis=dict(title="Predicted",tickfont=dict(color='#c2c8de')),
                yaxis=dict(title="Actual",tickfont=dict(color='#c2c8de')),
                height=380)
            fig.update_coloraxes(showscale=False)
            with col: st.plotly_chart(fig,use_container_width=True)

    with t3:
        c1,c2=st.columns(2)
        for col,model,nm,color in [(c1,rf,"Random Forest","#c9a84c"),(c2,xgb,"XGBoost","#4fc4cf")]:
            imp=pd.DataFrame({'feature':fcols,'importance':model.feature_importances_})
            imp=imp.sort_values('importance',ascending=True).tail(15)
            fig=go.Figure(go.Bar(x=imp.importance,y=imp.feature,orientation='h',
                marker=dict(color=imp.importance,
                    colorscale=[[0,'#1f253a'],[0.4,color],[1,color]],line_width=0),
                text=[f"{v:.3f}" for v in imp.importance],textposition='outside',
                textfont=dict(color='#c2c8de',size=10)))
            fig.update_layout(**pl(),height=500,title=f"Feature Importance — {nm}")
            with col: st.plotly_chart(fig,use_container_width=True)


# ════════════════════════════════════════════════════════════
#  ◎  PREDICT
# ════════════════════════════════════════════════════════════
elif page == "Predict":
    st.markdown(page_title("Live Performance Prediction","Performance",
        "Enter employee details for instant ML prediction with HR recommendations"), unsafe_allow_html=True)

    model_choice=st.radio("Model",["Random Forest 🌲","XGBoost ⚡"],horizontal=True)
    sel=rf if "Forest" in model_choice else xgb

    st.markdown(sec("Employee Details"), unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    with c1:
        st.markdown("<div style='font-size:0.68rem;color:#7b849e;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.7rem;font-family:Space Mono,monospace;'>Demographics</div>",unsafe_allow_html=True)
        age=st.slider("Age",22,60,30)
        exp=st.slider("Experience (Yrs)",0,35,5)
        dept=st.selectbox("Department",['Engineering','Sales','Marketing','HR','Finance','Operations'])
        sal=st.number_input("Salary ($)",25000,200000,55000,1000)
    with c2:
        st.markdown("<div style='font-size:0.68rem;color:#7b849e;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.7rem;font-family:Space Mono,monospace;'>Engagement</div>",unsafe_allow_html=True)
        trn=st.slider("Training Hours",0,80,40)
        sat=st.slider("Satisfaction (1-10)",1.0,10.0,7.0,0.1)
        att=st.slider("Attendance %",60.0,100.0,90.0,0.5)
    with c3:
        st.markdown("<div style='font-size:0.68rem;color:#7b849e;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.7rem;font-family:Space Mono,monospace;'>Output</div>",unsafe_allow_html=True)
        proj=st.slider("Projects Completed",1,20,8)
        moh=st.slider("Avg Monthly Hours",140,310,200)
        prom=st.selectbox("Promotions (5 Yrs)",[0,1,2,3])
        acc_=st.selectbox("Work Accident",[0,1],format_func=lambda x:"Yes" if x else "No")

    st.markdown("<br>",unsafe_allow_html=True)
    if st.button("◎  RUN PREDICTION"):
        dlist=['Engineering','Finance','HR','Marketing','Operations','Sales']
        inp={'age':age,'experience_years':exp,'salary':sal,'training_hours':trn,
             'satisfaction_score':sat,'attendance_pct':att,'projects_completed':proj,
             'avg_monthly_hours':moh,'promotions_last_5yrs':prom,'work_accidents':acc_}
        for d2 in dlist: inp[f'dept_{d2}']=1 if dept==d2 else 0
        idf=pd.DataFrame([inp])
        idf['engagement']     =(sat/10*50)+(att/100*50)
        idf['salary_per_exp'] =sal/(exp+1)
        idf['train_per_proj'] =trn/(proj+1)
        idf['overwork']       =int(moh>250)
        idf['career_momentum']=prom/(exp+1)
        for c in fcols:
            if c not in idf.columns: idf[c]=0
        idf=idf[fcols]
        isc=scaler.transform(idf)
        pred=sel.predict(isc)[0]
        prob=sel.predict_proba(isc)[0]
        label=rlmap[pred]
        accent={"High":"#4ecb8d","Medium":"#f5a623","Low":"#e05c7a"}[label]
        icon_={"High":"▲","Medium":"◉","Low":"▼"}[label]
        action={"High":"⭐ Strong promotion candidate. Fast-track for leadership visibility.",
                "Medium":"📈 Assign stretch goals. Enroll in upskilling programs.",
                "Low":"🔴 Immediate performance improvement plan. Schedule coaching."}[label]

        r1,r2=st.columns([1,2])
        with r1:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#111525,#181d2e);
              border:1px solid {accent}33;border-top:3px solid {accent};
              border-radius:18px;padding:2rem;text-align:center;">
              <div style="font-family:'Space Mono',monospace;font-size:0.65rem;color:#7b849e;
                text-transform:uppercase;letter-spacing:0.12em;margin-bottom:1rem;">Prediction Result</div>
              <div style="font-size:3.5rem;margin-bottom:0.4rem;">{icon_}</div>
              <div style="font-family:'Space Grotesk',sans-serif;font-size:2rem;font-weight:800;
                color:{accent};letter-spacing:0.04em;">{label.upper()}</div>
              <div style="font-size:0.75rem;color:#454d62;margin:0.4rem 0 1.5rem;">Performance Level</div>
              <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:0.45rem;margin-bottom:1.5rem;">
                {"".join([f'''<div style="background:rgba({r},0.08);border:1px solid rgba({r},0.2);
                  border-radius:9px;padding:0.55rem 0.2rem;">
                  <div style="font-size:0.58rem;color:#{hx};text-transform:uppercase;
                    letter-spacing:0.07em;margin-bottom:0.2rem;">{lb}</div>
                  <div style="font-family:'Space Mono',monospace;font-size:0.95rem;
                    font-weight:600;color:#{hx};">{v*100:.1f}%</div>
                </div>'''
                for lb,hx,r,v in [("Low","e05c7a","224,92,122",prob[0]),
                                   ("Med","f5a623","245,166,35",prob[1]),
                                   ("High","4ecb8d","78,203,141",prob[2])]
                ])}
              </div>
              <div style="background:{accent}11;border:1px solid {accent}33;border-left:3px solid {accent};
                border-radius:10px;padding:0.9rem;text-align:left;">
                <div style="font-size:0.8rem;color:{accent};line-height:1.65;">{action}</div>
              </div>
            </div>""", unsafe_allow_html=True)

        with r2:
            # Gauge
            fig_g=go.Figure(go.Indicator(
                mode="gauge+number",value=prob[pred]*100,
                title=dict(text="Model Confidence",font=dict(color='#c2c8de',size=13,family='Space Grotesk')),
                number=dict(suffix="%",font=dict(color=accent,size=44,family='Space Grotesk')),
                gauge=dict(
                    axis=dict(range=[0,100],tickfont=dict(color='#7b849e',size=9)),
                    bar=dict(color=accent,thickness=0.28),bgcolor='rgba(0,0,0,0)',
                    bordercolor='rgba(255,255,255,0.07)',
                    steps=[dict(range=[0,40],color='rgba(224,92,122,0.12)'),
                           dict(range=[40,70],color='rgba(245,166,35,0.12)'),
                           dict(range=[70,100],color='rgba(78,203,141,0.12)')],
                    threshold=dict(line=dict(color=accent,width=3),value=prob[pred]*100)
                )
            ))
            fig_g.update_layout(**{k:v for k,v in pl().items() if k not in['xaxis','yaxis']},height=270)
            st.plotly_chart(fig_g,use_container_width=True)

            # Probability bars
            fig_b=go.Figure(go.Bar(
                x=['Low','Medium','High'],y=prob*100,
                marker=dict(color=['#e05c7a','#f5a623','#4ecb8d'],line_width=0,opacity=0.88),
                text=[f"{p*100:.1f}%" for p in prob],textposition='outside',
                textfont=dict(color='#c2c8de',size=12,family='Space Mono'),
            ))
            fig_b.update_layout(**{k:v for k,v in pl().items() if k not in['yaxis']},
                                height=260,title="Class Probabilities",
                                yaxis=dict(**PL['yaxis'],range=[0,118]))
            st.plotly_chart(fig_b,use_container_width=True)

            # Factor bars
            imp2=dict(zip(fcols,sel.feature_importances_))
            top5=sorted(imp2.items(),key=lambda x:x[1],reverse=True)[:5]
            st.markdown("<div style='font-size:0.68rem;color:#7b849e;text-transform:uppercase;letter-spacing:0.1em;font-family:Space Mono,monospace;margin-bottom:0.6rem;'>Key Drivers</div>",unsafe_allow_html=True)
            mx=max(v for _,v in top5)
            for feat,val in top5:
                pct=int(val/mx*100)
                st.markdown(f"""
                <div style="margin-bottom:0.55rem;">
                  <div style="display:flex;justify-content:space-between;
                    font-size:0.78rem;color:#c2c8de;margin-bottom:0.22rem;">
                    <span>{feat}</span>
                    <span style="font-family:'Space Mono',monospace;color:#c9a84c;">{val:.3f}</span>
                  </div>
                  <div style="height:4px;background:rgba(255,255,255,0.05);border-radius:2px;overflow:hidden;">
                    <div style="height:100%;width:{pct}%;
                      background:linear-gradient(90deg,#c9a84c,#4fc4cf);border-radius:2px;"></div>
                  </div>
                </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  ◆  HR INSIGHTS
# ════════════════════════════════════════════════════════════
elif page == "HR Insights":
    st.markdown(page_title("HR Intelligence Report","Intelligence",
        "Batch predictions · Workforce analytics · Actionable recommendations"), unsafe_allow_html=True)

    dp=df.copy()
    dp=pd.get_dummies(dp,columns=['department'],prefix='dept')
    dp['target']         =dp['performance_label'].map(lmap)
    dp['engagement']     =(dp['satisfaction_score']/10*50)+(dp['attendance_pct']/100*50)
    dp['salary_per_exp'] =dp['salary']/(dp['experience_years']+1)
    dp['train_per_proj'] =dp['training_hours']/(dp['projects_completed']+1)
    dp['overwork']       =(dp['avg_monthly_hours']>250).astype(int)
    dp['career_momentum']=dp['promotions_last_5yrs']/(dp['experience_years']+1)
    Xf=dp[fcols].fillna(dp[fcols].median())
    df['predicted']=[rlmap[p] for p in xgb.predict(scaler.transform(Xf))]

    pr=(df.predicted=='High').sum()
    np_=(df.predicted=='Low').sum()
    risk=df[(df.predicted=='Low')&(df.satisfaction_score<4)].shape[0]
    tdept=df[df.predicted=='High']['department'].mode()[0]

    c1,c2,c3,c4=st.columns(4)
    c1.markdown(kpi("Promotion Ready",str(pr),f"{pr/len(df)*100:.1f}% of workforce","#4ecb8d"),unsafe_allow_html=True)
    c2.markdown(kpi("Need Action Plan",str(np_),f"{np_/len(df)*100:.1f}% of workforce","#e05c7a"),unsafe_allow_html=True)
    c3.markdown(kpi("Attrition Risk",str(risk),"Low perf + Low satisfaction","#f5a623"),unsafe_allow_html=True)
    c4.markdown(kpi("Top Department",tdept,"Highest high-performer rate","#4fc4cf"),unsafe_allow_html=True)

    t1,t2,t3=st.tabs(["Action Lists","Workforce Analytics","Recommendations"])

    with t1:
        c1,c2=st.columns(2)
        with c1:
            st.markdown("<div style='font-family:'Space Grotesk',sans-serif;font-size:0.8rem;font-weight:700;color:#4ecb8d;margin-bottom:0.6rem;text-transform:uppercase;letter-spacing:0.05em;'>▲ Promotion Candidates</div>",unsafe_allow_html=True)
            st.dataframe(df[df.predicted=='High'][['employee_id','department','experience_years','salary','performance_score']].sort_values('performance_score',ascending=False).head(10),use_container_width=True,height=340)
        with c2:
            st.markdown("<div style='font-family:'Space Grotesk',sans-serif;font-size:0.8rem;font-weight:700;color:#e05c7a;margin-bottom:0.6rem;text-transform:uppercase;letter-spacing:0.05em;'>▼ Immediate Intervention</div>",unsafe_allow_html=True)
            st.dataframe(df[df.predicted=='Low'][['employee_id','department','satisfaction_score','training_hours','performance_score']].sort_values('performance_score').head(10),use_container_width=True,height=340)

    with t2:
        c1,c2=st.columns(2)
        with c1:
            pc2=df.predicted.value_counts().reset_index(); pc2.columns=['label','count']; pc2['type']='Predicted'
            ac2=df.performance_label.value_counts().reset_index(); ac2.columns=['label','count']; ac2['type']='Actual'
            fig=px.bar(pd.concat([ac2,pc2]),x='label',y='count',color='type',barmode='group',
                       color_discrete_sequence=['#4fc4cf','#c9a84c'],
                       title="Actual vs Predicted",
                       category_orders={'label':['High','Medium','Low']})
            fig.update_traces(marker_line_width=0,opacity=0.9)
            fig.update_layout(**pl())
            st.plotly_chart(fig,use_container_width=True)
        with c2:
            dpred=df.groupby(['department','predicted']).size().reset_index(name='count')
            fig2=px.bar(dpred,x='department',y='count',color='predicted',barmode='stack',
                        color_discrete_map=PC,title="Predicted Performance by Dept")
            fig2.update_traces(marker_line_width=0)
            fig2.update_layout(**pl(),xaxis_tickangle=-30)
            st.plotly_chart(fig2,use_container_width=True)
        samp2=df.sample(min(400,len(df)),random_state=7)
        fig3=px.scatter(samp2,x='satisfaction_score',y='performance_score',
                        color='predicted',color_discrete_map=PC,
                        size='training_hours',size_max=14,opacity=0.72,
                        title="Satisfaction vs Performance (sized by training hours)",
                        hover_data=['employee_id','department'])
        fig3.update_traces(marker_line_width=0)
        fig3.update_layout(**pl())
        st.plotly_chart(fig3,use_container_width=True)

    with t3:
        recs=[
            ("#c9a84c","▲","Training ROI",
             f"Low performers average {df[df.predicted=='Low']['training_hours'].mean():.0f} training hrs vs "
             f"{df[df.predicted=='High']['training_hours'].mean():.0f} for high performers. "
             f"Increase training investment for bottom {np_} employees by 40%+."),
            ("#e05c7a","⚠","Attrition Red Zone",
             f"{risk} employees have both low predicted performance AND satisfaction below 4.0. "
             "These are highest flight-risk. Schedule manager 1-on-1s within 2 weeks."),
            ("#4ecb8d","★","Reward Top Talent",
             f"{pr} employees are high-performance candidates. "
             "High performers leave if unrecognized — create visible promotion timelines and milestone rewards."),
            ("#4fc4cf","◈","Department Strategy",
             f"Engineering shows the highest avg performance. "
             f"{df[df.predicted=='Low']['department'].mode()[0]} needs structural review — "
             "workload balance, team leads, and satisfaction initiatives."),
            ("#a78bfa","◎","Salary Alignment",
             f"Top performers earn avg ${df[df.predicted=='High']['salary'].mean():,.0f} vs "
             f"${df[df.predicted=='Low']['salary'].mean():,.0f} for low performers. "
             "Review compensation bands to ensure performance-linked pay structures."),
        ]
        for color,icon,title,body in recs:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#111525,#181d2e);
              border:1px solid rgba(255,255,255,0.06);border-left:3px solid {color};
              border-radius:14px;padding:1.3rem 1.5rem;margin-bottom:0.8rem;
              display:flex;align-items:flex-start;gap:1.2rem;">
              <div style="font-size:1.3rem;color:{color};flex-shrink:0;margin-top:0.1rem;">{icon}</div>
              <div>
                <div style="font-family:'Space Grotesk',sans-serif;font-size:0.85rem;font-weight:700;
                  color:#f0f2fa;margin-bottom:0.35rem;">{title}</div>
                <div style="font-size:0.81rem;color:#7b849e;line-height:1.7;">{body}</div>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>",unsafe_allow_html=True)
        st.download_button("⬇ Download Predictions CSV",
            df[['employee_id','department','performance_label','predicted','performance_score']].to_csv(index=False),
            "predictions.csv","text/csv")
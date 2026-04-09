"""
app.py
──────
Streamlit dashboard for the Customer Churn Prediction Pipeline.

Pages
─────
  🏠 Overview          – project summary & live metrics
  🔍 Single Prediction – predict churn for one customer
  📊 Batch Prediction  – upload CSV → download predictions
  📈 Model Comparison  – compare all 4 trained models
  🧠 SHAP Explainability – feature-importance plots
  📋 MLflow Runs        – live run table from MLflow

Run
───
  streamlit run app/app.py
"""

import os, sys, json, pathlib, warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import requests
from streamlit_lottie import st_lottie
import streamlit.components.v1 as components

warnings.filterwarnings("ignore")

# ── Resolve project root regardless of CWD ───────────────────────────────────
APP_DIR  = pathlib.Path(__file__).resolve().parent
BASE_DIR = APP_DIR.parent
sys.path.insert(0, str(BASE_DIR / "src"))

ART_DIR   = BASE_DIR / "artifacts"
SHAP_DIR  = ART_DIR  / "shap"
MODEL_DIR = BASE_DIR / "models"
PROC_DIR  = BASE_DIR / "data" / "processed"
RES_JSON  = ART_DIR  / "results.json"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnFlow · ML Dashboard",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background-color: #72A0C1; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.15) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.3) !important;
}
[data-testid="stSidebar"] * { color: #4B5563 !important; }
[data-testid="stSidebar"] .stRadio label { 
    font-size: 15px; font-weight: 500; padding: 10px 12px;
    border-radius: 6px; transition: all 0.2s ease;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: #F3F4F6;
    color: #111827 !important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label[data-checked="true"] {
    background: #EEF2FF;
    border-left: 3px solid #635BFF; color: #635BFF !important;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.3);
    border-radius: 12px;
    padding: 20px 24px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    transition: all 0.3s ease;
}
[data-testid="metric-container"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    border-color: #D1D5DB;
}
[data-testid="metric-container"] label { color: #6B7280 !important; font-size: 13px !important; letter-spacing: 0.5px; text-transform: uppercase; font-weight: 600; }
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #111827 !important; font-size: 32px !important; font-weight: 700 !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { gap: 8px; background: transparent; }
.stTabs [data-baseweb="tab"] {
    background: #F3F4F6; border-radius: 8px; color: #6B7280;
    padding: 8px 16px; border: 1px solid transparent;
    font-weight: 500; font-size: 14px; transition: all 0.2s ease;
}
.stTabs [data-baseweb="tab"]:hover {
    background: #E5E7EB; color: #111827;
}
.stTabs [aria-selected="true"] {
    background: #FFFFFF !important;
    color: #635BFF !important; border-color: #E5E7EB !important;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
}

/* ── Buttons ── */
.stButton > button {
    background: #635BFF;
    color: white; border: none; border-radius: 8px;
    padding: 10px 24px; font-weight: 600; font-size: 14px;
    transition: all 0.2s ease; box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
}
.stButton > button:hover { 
    transform: translateY(-1px); box-shadow: 0 4px 6px -1px rgba(99, 91, 255, 0.4); 
    background: #4F46E5; color: white;
}

/* ── Cards ── */
.glass-card {
    background: rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.3);
    border-radius: 12px; padding: 24px; margin: 16px 0;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    transition: all 0.3s ease;
}
.glass-card:hover { box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1); }

/* ── Churn badges ── */
.churn-yes { 
    background: #FEE2E2;
    color: #991B1B; padding: 8px 24px; border-radius: 9999px;
    font-weight: 700; font-size: 16px; display: inline-block;
}
.churn-no { 
    background: #D1FAE5;
    color: #065F46; padding: 8px 24px; border-radius: 9999px;
    font-weight: 700; font-size: 16px; display: inline-block;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #D1D5DB; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #9CA3AF; }

/* ── Headings ── */
h1, h2, h3, h4 { color: #111827 !important; font-weight: 700 !important; letter-spacing: -0.5px; }
hr { border-color: #E5E7EB !important; margin: 24px 0; }

/* ── Advanced Animated Entrances (GSAP-like) ── */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}
[data-testid="metric-container"], .glass-card, [data-testid="stDataFrame"], .stPlotlyChart {
    animation: fadeInUp 0.7s cubic-bezier(0.16, 1, 0.3, 1) both;
}
[data-testid="metric-container"]:nth-child(1) { animation-delay: 0.1s; }
[data-testid="metric-container"]:nth-child(2) { animation-delay: 0.2s; }
[data-testid="metric-container"]:nth-child(3) { animation-delay: 0.3s; }
[data-testid="metric-container"]:nth-child(4) { animation-delay: 0.4s; }
[data-testid="metric-container"]:nth-child(5) { animation-delay: 0.5s; }
html { scroll-behavior: smooth; }
</style>

<!-- Fullscreen Spline Background -->
<iframe src="https://my.spline.design/particles-9dc55d64821815db976fc93175c5e888/" style="position: fixed; inset: 0; width: 100vw; height: 100vh; border: none; z-index: -99; opacity: 0.6; pointer-events: auto;"></iframe>

""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(name: str):
    path = MODEL_DIR / f"{name}.joblib"
    return joblib.load(path) if path.exists() else None


@st.cache_resource
def load_preprocessor():
    path = MODEL_DIR / "preprocessor.joblib"
    return joblib.load(path) if path.exists() else None


@st.cache_data
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


@st.cache_data
def load_results() -> dict:
    if RES_JSON.exists():
        with open(RES_JSON) as f:
            return json.load(f)
    return {}


@st.cache_data
def load_features() -> list[str]:
    p = PROC_DIR / "feature_names.txt"
    return p.read_text().split("\n") if p.exists() else []


def results_dataframe(results: dict) -> pd.DataFrame:
    rows = []
    for m, r in results.items():
        rows.append({
            "Model"       : m.replace("_", " "),
            "CV AUC"      : round(r["cv_auc_mean"], 4),
            "AUC Std"     : round(r["cv_auc_std"],  4),
            "CV F1"       : round(r["cv_f1_mean"],  4),
            "F1 Std"      : round(r["cv_f1_std"],   4),
            "CV Accuracy" : round(r["cv_accuracy_mean"], 4),
            "Precision"   : round(r["cv_precision_mean"], 4),
            "Recall"      : round(r["cv_recall_mean"], 4),
        })
    return pd.DataFrame(rows).sort_values("CV AUC", ascending=False).reset_index(drop=True)


MODEL_NAMES = ["Logistic_Regression", "Random_Forest", "XGBoost", "LightGBM"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:20px 0 10px'>
        <div style='font-size:48px'>🔮</div>
        <div style='font-size:20px; font-weight:700; letter-spacing:1px'>ChurnFlow</div>
        <div style='font-size:12px; color:#8888cc; margin-top:4px'>ML Pipeline Dashboard</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    page = st.radio(
        "Navigate",
        ["🏠  Overview", "🔍  Single Prediction", "📊  Batch Prediction",
         "📈  Model Comparison", "🧠  SHAP Explainability", "📋  MLflow Runs"],
        label_visibility="collapsed"
    )

    st.divider()
    results = load_results()
    if results:
        best = max(results, key=lambda k: results[k]["cv_auc_mean"])
        st.markdown(f"""
        <div style='text-align:center'>
            <div style='font-size:12px; color:#6B7280; font-weight:600; letter-spacing: 0.5px;'>🏆 BEST MODEL</div>
            <div style='font-size:16px; font-weight:700; color:#635BFF;'>
                {best.replace("_"," ")}
            </div>
            <div style='font-size:22px; font-weight:800; color:#10B981; margin-top:4px'>
                AUC {results[best]["cv_auc_mean"]:.4f}
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:12px; color:#4a5568; text-align:center'>Built by Nikhil Yadav · BIT Durg</div>", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE — OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
if "Overview" in page:
    st.markdown("## 🔄 Customer Churn Prediction Pipeline")
    st.markdown("*End-to-end ML pipeline with MLflow tracking, SHAP explainability & model registry*")
    st.divider()

    # KPI row
    results = load_results()
    if results:
        df_r = results_dataframe(results)
        best_row = df_r.iloc[0]
        # Serialization for React + GSAP
        react_data = json.dumps({
            "model": best_row["Model"],
            "auc": f"{best_row['CV AUC']:.4f}",
            "f1": f"{best_row['CV F1']:.4f}",
            "acc": f"{best_row['CV Accuracy']:.4f}",
            "trained": len(results)
        })

        html_code = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <script src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
            <script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
            <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.2/gsap.min.js"></script>
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
                body {{ font-family: 'Inter', sans-serif; margin: 0; padding: 10px 0; background: transparent; }}
                .kpi-container {{ display: flex; gap: 16px; justify-content: space-between; overflow: hidden; height: 120px; }}
                .kpi-card {{
                    background: rgba(255, 255, 255, 0.15);
                    backdrop-filter: blur(20px);
                    -webkit-backdrop-filter: blur(20px);
                    border: 1px solid rgba(255, 255, 255, 0.3);
                    border-radius: 12px;
                    padding: 20px;
                    width: 100%;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    opacity: 0;
                    transform: translateY(30px);
                }}
                .kpi-title {{ color: #6B7280; font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px; }}
                .kpi-value {{ color: #111827; font-size: 28px; font-weight: 700; }}
            </style>
        </head>
        <body>
            <div id="react-root"></div>
            <script type="text/babel">
                const data = {react_data};
                
                function KpiDashboard() {{
                    React.useEffect(() => {{
                        gsap.to(".kpi-card", {{
                            y: 0,
                            opacity: 1,
                            duration: 0.8,
                            stagger: 0.1,
                            ease: "back.out(1.7)"
                        }});
                    }}, []);

                    return (
                        <div className="kpi-container">
                            <div className="kpi-card"><div className="kpi-title">🏆 Best Model</div><div className="kpi-value">{{data.model}}</div></div>
                            <div className="kpi-card"><div className="kpi-title">📊 Best AUC</div><div className="kpi-value">{{data.auc}}</div></div>
                            <div className="kpi-card"><div className="kpi-title">🎯 Best F1</div><div className="kpi-value">{{data.f1}}</div></div>
                            <div className="kpi-card"><div className="kpi-title">✅ Best Acc</div><div className="kpi-value">{{data.acc}}</div></div>
                            <div className="kpi-card"><div className="kpi-title">🔁 Trained</div><div className="kpi-value">{{data.trained}}</div></div>
                        </div>
                    );
                }}

                const root = ReactDOM.createRoot(document.getElementById('react-root'));
                root.render(<KpiDashboard />);
            </script>
        </body>
        </html>
        """
        components.html(html_code, height=140)
    else:
        st.info("⚠️  No training results found. Run `python src/train.py` to populate this dashboard.")

    st.markdown("<br>", unsafe_allow_html=True)

    # Pipeline architecture diagram
    st.markdown("### 🏗️ Pipeline Architecture")
    col_diag, col_stack = st.columns([3, 2])

    with col_diag:
        steps = [
            ("📥", "Data",       "#6B7280", 0, 1),
            ("🧹", "Preprocess", "#8B5CF6", 1, 1),
            ("🗄️", "Features",   "#3B82F6", 2, 1),
            ("⚖️", "SMOTE",      "#10B981", 3, 1),
            ("🤖", "Models",     "#635BFF", 4, 1),
            ("📊", "Autolog",    "#F59E0B", 4, 0),
            ("🧠", "SHAP",       "#EC4899", 3, 0),
            ("🏭", "Registry",   "#635BFF", 2, 0),
            ("🖥️", "UI",         "#111827", 1, 0),
        ]
        fig = go.Figure()
        for i, (icon, name, color, x, y) in enumerate(steps):
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode="markers+text",
                marker=dict(size=45, color=color, symbol="square", line=dict(color="#FFFFFF", width=2)),
                text=[f"<b>{icon}<br>{name}</b>"],
                textposition="top center" if y == 1 else "bottom center",
                textfont=dict(size=11, color="#111827"),
            ))
            # Draw arrows to next step
            if i < len(steps) - 1:
                nx, ny = steps[i+1][3], steps[i+1][4]
                # Arrow style based on direction
                arrow = "→" if nx > x else "←" if nx < x else "↓"
                # Midpoint for arrow
                ax, ay = (x + nx) / 2, (y + ny) / 2
                if nx == x: # It's the drop-down arrow
                   ax += 0.1 
                fig.add_annotation(
                    x=ax, y=ay, text=arrow, showarrow=False,
                    font=dict(size=22, color="#6B7280"), xanchor="center"
                )
                
        fig.update_layout(
            height=280, margin=dict(l=20, r=20, t=30, b=30),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[-0.5, 4.5]),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[-0.5, 1.5]),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_stack:
        st.markdown("""
        <div class='glass-card'>
            <b style='color:#635BFF; font-size:16px; letter-spacing:0.5px;'>TECH STACK</b><br><br>
            🐍 Python · Pandas · NumPy<br>
            🤖 Scikit-learn · XGBoost · LightGBM<br>
            ⚖️ imbalanced-learn (SMOTE)<br>
            📊 MLflow Tracking + Registry<br>
            🧠 SHAP Explainability<br>
            🖥️ Streamlit Dashboard<br>
            🐳 Docker Deployment
        </div>
        """, unsafe_allow_html=True)

    # Model comparison mini-chart
    if results:
        st.markdown("### 📊 Quick Model Comparison")
        df_r = results_dataframe(results)
        fig2 = make_subplots(rows=1, cols=3, subplot_titles=["AUC-ROC", "F1 Score", "Accuracy"])
        colors = ["#635BFF", "#10B981", "#8B5CF6", "#3B82F6"]
        for i, metric in enumerate(["CV AUC", "CV F1", "CV Accuracy"]):
            fig2.add_trace(go.Bar(
                x=df_r["Model"], y=df_r[metric],
                marker_color=colors, showlegend=False,
                text=df_r[metric].round(3), textposition="outside",
                textfont=dict(color="#111827", size=11)
            ), row=1, col=i+1)
        fig2.update_layout(
            height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#111827"), margin=dict(t=40, b=10),
        )
        fig2.update_xaxes(tickangle=-20, tickfont=dict(size=10, color="#111827"))
        fig2.update_yaxes(gridcolor="#E5E7EB", range=[0.5, 1.0], tickfont=dict(color="#111827"))
        st.plotly_chart(fig2, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE — SINGLE PREDICTION
# ═════════════════════════════════════════════════════════════════════════════
elif "Single Prediction" in page:
    st.markdown("## 🔍 Single Customer Prediction")
    st.markdown("Fill in customer details to predict churn probability.")
    st.divider()

    preprocessor = load_preprocessor()
    results = load_results()

    if not results:
        st.warning("No trained models found. Run `python src/train.py` first.")
    else:
        model_choice = st.selectbox(
            "Select Model",
            [m.replace("_", " ") for m in MODEL_NAMES if (MODEL_DIR / f"{m}.joblib").exists()],
            index=0
        )
        model_key  = model_choice.replace(" ", "_")
        model      = load_model(model_key)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 👤 Customer Profile")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**📋 Account Info**")
            gender          = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen  = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner         = st.selectbox("Partner", ["Yes", "No"])
            dependents      = st.selectbox("Dependents", ["No", "Yes"])
            tenure          = st.slider("Tenure (months)", 0, 72, 12)
            contract        = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

        with c2:
            st.markdown("**📱 Services**")
            phone_service   = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines  = st.selectbox("Multiple Lines", ["No", "Yes"])
            internet_service= st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
            online_security = st.selectbox("Online Security", ["No", "Yes"])
            online_backup   = st.selectbox("Online Backup", ["No", "Yes"])
            device_prot     = st.selectbox("Device Protection", ["No", "Yes"])

        with c3:
            st.markdown("**💳 Billing**")
            tech_support    = st.selectbox("Tech Support", ["No", "Yes"])
            streaming_tv    = st.selectbox("Streaming TV", ["No", "Yes"])
            streaming_movies= st.selectbox("Streaming Movies", ["No", "Yes"])
            paperless       = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method  = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            monthly_charges = st.number_input("Monthly Charges ($)", 18.0, 120.0, 65.0, 0.5)
            total_charges   = st.number_input("Total Charges ($)", 0.0, 9000.0,
                                              float(monthly_charges * tenure), 1.0)

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🔮 Predict Churn"):
            raw_row = pd.DataFrame([{
                "gender"           : gender,
                "SeniorCitizen"    : 1 if senior_citizen == "Yes" else 0,
                "Partner"          : partner,
                "Dependents"       : dependents,
                "tenure"           : tenure,
                "PhoneService"     : phone_service,
                "MultipleLines"    : multiple_lines,
                "InternetService"  : internet_service,
                "OnlineSecurity"   : online_security,
                "OnlineBackup"     : online_backup,
                "DeviceProtection" : device_prot,
                "TechSupport"      : tech_support,
                "StreamingTV"      : streaming_tv,
                "StreamingMovies"  : streaming_movies,
                "Contract"         : contract,
                "PaperlessBilling" : paperless,
                "PaymentMethod"    : payment_method,
                "MonthlyCharges"   : monthly_charges,
                "TotalCharges"     : total_charges,
            }])

            # Feature engineering (mirrors preprocess.py)
            raw_row["tenure_group"]    = pd.cut(
                raw_row["tenure"],
                bins=[0,12,24,48,60,np.inf],
                labels=["0-1yr","1-2yr","2-4yr","4-5yr","5+yr"]
            ).astype(str)
            raw_row["revenue_per_month"] = raw_row["TotalCharges"] / (raw_row["tenure"] + 1)
            raw_row["high_value"]        = (raw_row["MonthlyCharges"] > 65.0).astype(int)
            svc_cols = ["OnlineSecurity","OnlineBackup","DeviceProtection",
                        "TechSupport","StreamingTV","StreamingMovies"]
            raw_row["num_services"] = raw_row[svc_cols].apply(
                lambda r: (r == "Yes").sum(), axis=1
            )

            if preprocessor:
                X_in = preprocessor.transform(raw_row)
            else:
                st.error("Preprocessor not found. Run preprocess.py first.")
                st.stop()

            prob   = model.predict_proba(X_in)[0][1]
            label  = "YES — Likely to Churn" if prob >= 0.5 else "NO — Likely to Stay"
            badge  = "churn-yes" if prob >= 0.5 else "churn-no"

            st.divider()
            r1, r2, r3 = st.columns([2, 1, 1])
            with r1:
                st.markdown(f"### Prediction Result")
                st.markdown(f"<div class='{badge}'>{label}</div>", unsafe_allow_html=True)
            with r2:
                st.metric("Churn Probability", f"{prob:.1%}")
            with r3:
                st.metric("Retention Probability", f"{1-prob:.1%}")

            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob * 100,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Churn Risk Score", "font": {"color": "#111827", "size": 18}},
                number={"suffix": "%", "font": {"color": "#111827", "size": 32}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#111827"},
                    "bar":  {"color": "#FF0844" if prob >= 0.5 else "#00B4DB"},
                    "steps": [
                        {"range": [0, 30],  "color": "rgba(0,180,219,0.2)"},
                        {"range": [30, 60], "color": "rgba(250,208,44,0.2)"},
                        {"range": [60, 100],"color": "rgba(255,8,68,0.2)"},
                    ],
                    "threshold": {"line": {"color": "#991B1B", "width": 3}, "value": 50}
                }
            ))
            fig_gauge.update_layout(
                height=280, paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#111827"), margin=dict(t=30, b=10)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE — BATCH PREDICTION
# ═════════════════════════════════════════════════════════════════════════════
elif "Batch Prediction" in page:
    st.markdown("## 📊 Batch Prediction")
    c1, c2 = st.columns([4, 1])
    with c1:
        st.markdown("Upload a CSV of customers (same schema as Telco dataset) to predict churn at scale.")
    with c2:
        lottie_file = load_lottieurl("https://lottie.host/80a221f1-332e-4375-9276-200742d4a77e/S2oWkO6aOo.json")
        if lottie_file:
            st_lottie(lottie_file, height=80, key="batch_lottie")
    st.divider()

    preprocessor = load_preprocessor()
    results      = load_results()

    if not results:
        st.warning("No trained models found. Run `python src/train.py` first.")
    else:
        model_choice = st.selectbox(
            "Select Model",
            [m.replace("_", " ") for m in MODEL_NAMES if (MODEL_DIR / f"{m}.joblib").exists()],
        )
        model = load_model(model_choice.replace(" ", "_"))

        uploaded = st.file_uploader("Upload CSV (Telco schema, without 'Churn' column)", type="csv")

        if uploaded:
            df_raw = pd.read_csv(uploaded)
            st.markdown(f"**Loaded {len(df_raw)} customers**")

            # Minimal cleaning
            df_in = df_raw.copy()
            df_in.drop(columns=["customerID", "Churn"], inplace=True, errors="ignore")
            df_in["TotalCharges"] = pd.to_numeric(df_in["TotalCharges"], errors="coerce")
            df_in["TotalCharges"].fillna(df_in["TotalCharges"].median(), inplace=True)
            df_in.replace({"No internet service": "No", "No phone service": "No"}, inplace=True)

            # Feature engineering
            df_in["tenure_group"]     = pd.cut(df_in["tenure"], bins=[0,12,24,48,60,np.inf],
                                                labels=["0-1yr","1-2yr","2-4yr","4-5yr","5+yr"]).astype(str)
            df_in["revenue_per_month"]= df_in["TotalCharges"] / (df_in["tenure"] + 1)
            df_in["high_value"]       = (df_in["MonthlyCharges"] > 65.0).astype(int)
            svc_cols = ["OnlineSecurity","OnlineBackup","DeviceProtection",
                        "TechSupport","StreamingTV","StreamingMovies"]
            df_in["num_services"]     = df_in[svc_cols].apply(lambda r: (r == "Yes").sum(), axis=1)

            with st.spinner("Running predictions…"):
                X_in      = preprocessor.transform(df_in)
                probs      = model.predict_proba(X_in)[:, 1]
                preds      = (probs >= 0.5).astype(int)

            df_out = df_raw.copy()
            df_out["Churn_Prediction"] = preds
            df_out["Churn_Probability"]= probs.round(4)
            df_out["Risk_Level"]       = pd.cut(probs, bins=[0,.3,.6,1.0],
                                                 labels=["Low","Medium","High"]).astype(str)

            st.divider()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Customers",   len(df_out))
            m2.metric("Predicted to Churn", int(preds.sum()))
            m3.metric("Churn Rate",         f"{preds.mean():.1%}")
            m4.metric("Avg Churn Prob",     f"{probs.mean():.1%}")

            # Risk distribution pie
            risk_counts = df_out["Risk_Level"].value_counts()
            fig_pie = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                color=risk_counts.index,
                color_discrete_map={"Low":"#00B4DB","Medium":"#FAD02C","High":"#FF0844"},
                title="Churn Risk Distribution"
            )
            fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#111827"))

            fig_hist = px.histogram(
                df_out, x="Churn_Probability", nbins=40,
                color_discrete_sequence=["#00F2FE"],
                title="Churn Probability Distribution"
            )
            fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                   font=dict(color="#111827"))
            fig_hist.update_xaxes(gridcolor="#E5E7EB")
            fig_hist.update_yaxes(gridcolor="#E5E7EB")

            pc, ph = st.columns(2)
            pc.plotly_chart(fig_pie,  use_container_width=True)
            ph.plotly_chart(fig_hist, use_container_width=True)

            st.dataframe(
                df_out[["Churn_Probability","Churn_Prediction","Risk_Level"] +
                        [c for c in df_out.columns if c not in
                         ["Churn_Prediction","Churn_Probability","Risk_Level"]]
                       ].head(200),
                use_container_width=True, height=350
            )

            csv_out = df_out.to_csv(index=False).encode()
            st.download_button("📥 Download Predictions CSV", csv_out,
                               "churn_predictions.csv", "text/csv")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE — MODEL COMPARISON
# ═════════════════════════════════════════════════════════════════════════════
elif "Model Comparison" in page:
    st.markdown("## 📈 Model Comparison")
    st.divider()

    results = load_results()
    if not results:
        st.warning("Run `python src/train.py` to generate results.")
    else:
        df_r = results_dataframe(results)

        # Leaderboard table
        st.markdown("### 🏆 Leaderboard")
        st.dataframe(
            df_r.style
                .background_gradient(subset=["CV AUC","CV F1","CV Accuracy"], cmap="Blues")
                .format({"CV AUC":"{:.4f}","AUC Std":"{:.4f}",
                         "CV F1":"{:.4f}","F1 Std":"{:.4f}",
                         "CV Accuracy":"{:.4f}","Precision":"{:.4f}","Recall":"{:.4f}"}),
            use_container_width=True, height=200
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📊 Metric Breakdown")

        metrics = ["CV AUC", "CV F1", "CV Accuracy", "Precision", "Recall"]
        fig = go.Figure()
        colors = ["#00F2FE", "#4FACFE", "#00B4DB", "#0083B0"]

        rgba_map = {
            "#00F2FE": "rgba(0,242,254,0.15)",
            "#4FACFE": "rgba(79,172,254,0.15)",
            "#00B4DB": "rgba(0,180,219,0.15)",
            "#0083B0": "rgba(0,131,176,0.15)",
        }

        for i, row in df_r.iterrows():
            # Convert hex colors to rgba for Plotly compatibility
            hex_color   = colors[i % len(colors)]
            fill_color  = rgba_map.get(hex_color, "rgba(0,242,254,0.15)")

            fig.add_trace(go.Scatterpolar(
                r=[row[m] for m in metrics],
                theta=metrics,
                fill="toself",
                name=row["Model"],
                line=dict(color=hex_color, width=2),
                fillcolor=fill_color
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0.5, 1.0], gridcolor="#E5E7EB",
                                tickfont=dict(color="#111827"), tickcolor="#111827"),
                angularaxis=dict(gridcolor="#E5E7EB",
                                 tickfont=dict(color="#111827"))
            ),
            showlegend=True,
            legend=dict(font=dict(color="#111827")),
            paper_bgcolor="rgba(0,0,0,0)",
            height=480,
            font=dict(color="#111827")
        )
        st.plotly_chart(fig, use_container_width=True)

        # AUC ± std error bars
        st.markdown("### 🎯 AUC with Cross-Validation Std")
        fig_eb = go.Figure()
        fig_eb.add_trace(go.Bar(
            x=df_r["Model"], y=df_r["CV AUC"],
            error_y=dict(type="data", array=df_r["AUC Std"].tolist(), visible=True,
                         color="#111827", thickness=2, width=8),
            marker_color=colors[:len(df_r)],
            text=df_r["CV AUC"].round(4), textposition="outside",
            textfont=dict(color="#111827")
        ))
        fig_eb.update_layout(
            height=380, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#111827"), yaxis=dict(range=[0.6, 1.0], gridcolor="#E5E7EB"),
            margin=dict(t=30)
        )
        st.plotly_chart(fig_eb, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE — SHAP EXPLAINABILITY
# ═════════════════════════════════════════════════════════════════════════════
elif "SHAP" in page:
    st.markdown("## 🧠 SHAP Explainability")
    st.markdown("Feature importance and individual prediction explanations via SHAP values.")
    st.divider()

    available_models = [
        m for m in MODEL_NAMES
        if (SHAP_DIR / f"shap_summary_{m}.png").exists()
    ]

    if not available_models:
        st.warning("No SHAP artefacts found. Run `python src/shap_report.py` first.")
    else:
        model_choice = st.selectbox("Select Model",
                                    [m.replace("_", " ") for m in available_models])
        mk = model_choice.replace(" ", "_")

        tab_sum, tab_bar, tab_wf = st.tabs(["🐝 Beeswarm Summary", "📊 Feature Importance Bar", "💧 Waterfall"])

        with tab_sum:
            st.markdown(f"#### SHAP Summary Plot — {model_choice}")
            st.markdown("Each dot = one customer. Color = feature value (red=high, blue=low). "
                        "X-axis = SHAP value impact on churn prediction.")
            img_path = SHAP_DIR / f"shap_summary_{mk}.png"
            if img_path.exists():
                st.image(str(img_path), use_column_width=True)

        with tab_bar:
            st.markdown(f"#### Mean |SHAP| Feature Importance — {model_choice}")
            st.markdown("Global importance: average absolute SHAP value across all customers.")
            img_path = SHAP_DIR / f"shap_bar_{mk}.png"
            if img_path.exists():
                st.image(str(img_path), use_column_width=True)

            # Also render interactive Plotly bar from CSV
            csv_path = SHAP_DIR / f"shap_values_{mk}.csv"
            if csv_path.exists():
                shap_df   = pd.read_csv(csv_path)
                mean_abs  = shap_df.abs().mean().sort_values(ascending=False).head(20)
                fig_shap  = px.bar(
                    x=mean_abs.values, y=mean_abs.index, orientation="h",
                    labels={"x": "Mean |SHAP Value|", "y": "Feature"},
                    title=f"Top 20 Features — {model_choice}",
                    color=mean_abs.values,
                    color_continuous_scale="Teal"
                )
                fig_shap.update_layout(
                    height=550, paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#111827"),
                    yaxis=dict(autorange="reversed"), coloraxis_showscale=False,
                    xaxis=dict(gridcolor="#E5E7EB"),
                )
                st.plotly_chart(fig_shap, use_container_width=True)

        with tab_wf:
            st.markdown(f"#### Waterfall Plot — {model_choice}")
            st.markdown("Single-prediction explanation: how each feature pushes the prediction "
                        "above/below the base rate.")
            img_path = SHAP_DIR / f"shap_waterfall_{mk}.png"
            if img_path.exists():
                st.image(str(img_path), use_column_width=True)
            else:
                st.info("Waterfall plot not generated for this model.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE — MLFLOW RUNS
# ═════════════════════════════════════════════════════════════════════════════
elif "MLflow Runs" in page:
    st.markdown("## 📋 MLflow Run Explorer")
    st.divider()

    results = load_results()
    if not results:
        st.warning("No results found. Run `python src/train.py` first.")
    else:
        st.markdown("### All Training Runs")

        df_r = results_dataframe(results)
        run_ids = {r["Model"].replace(" ","_"): results[r["Model"].replace(" ","_")]["run_id"]
                   for _, r in df_r.iterrows()}
        df_r["Run ID"] = df_r["Model"].str.replace(" ", "_").map(run_ids).str[:8] + "…"

        display_cols = ["Model", "Run ID", "CV AUC", "AUC Std", "CV F1",
                        "CV Accuracy", "Precision", "Recall"]
        st.dataframe(df_r[display_cols], use_container_width=True, height=220)

        best_name = df_r.iloc[0]["Model"].replace(" ", "_")
        best_info = results[best_name]

        st.markdown("### 🏆 Best Run Details")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class='glass-card'>
                <b>Model:</b> {best_name.replace('_',' ')}<br><br>
                <b>Full Run ID:</b><br>
                <code style='font-size:11px'>{best_info['run_id']}</code><br><br>
                <b>Experiment:</b> Customer-Churn-Prediction
            </div>
            """, unsafe_allow_html=True)
        with c2:
            metric_labels = {
                "cv_auc_mean":"CV AUC","cv_f1_mean":"CV F1",
                "cv_accuracy_mean":"CV Accuracy","cv_precision_mean":"Precision",
                "cv_recall_mean":"Recall","train_auc":"Train AUC"
            }
            for k, label in metric_labels.items():
                if k in best_info:
                    st.metric(label, f"{best_info[k]:.4f}")

        st.info("💡 For the full interactive MLflow UI (run history, parameter plots, artefact browser), "
                "run: `mlflow ui --backend-store-uri sqlite:///mlflow.db` in your terminal.")
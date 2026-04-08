import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import mlflow
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📊",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load('models/best_model.pkl')

@st.cache_data
def load_data():
    return pd.read_csv('data/processed/features.csv')

model = load_model()
df    = load_data()

# ── Sidebar ──────────────────────────────
st.sidebar.title("📊 Churn Predictor")
page = st.sidebar.radio("Navigate",
    ["Dashboard", "Predict Customer", "SHAP Insights"])

# ── Dashboard ────────────────────────────
if page == "Dashboard":
    st.title("Customer Churn Dashboard")

    total     = len(df)
    churned   = df['Churn'].sum()
    retained  = total - churned
    churn_pct = round(churned / total * 100, 1)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{total:,}")
    col2.metric("Churned",         f"{churned:,}")
    col3.metric("Retained",        f"{retained:,}")
    col4.metric("Churn Rate",      f"{churn_pct}%",
                delta=f"-{100-churn_pct}% retained")

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Churn Distribution")
        fig = px.pie(
            values=[retained, churned],
            names=['Retained', 'Churned'],
            color_discrete_sequence=['#2196F3', '#F44336']
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Feature Correlations with Churn")
        corr = df.corr()['Churn'].drop('Churn')\
                  .sort_values(ascending=False).head(10)
        fig2 = px.bar(
            x=corr.values, y=corr.index,
            orientation='h',
            color=corr.values,
            color_continuous_scale=['#2196F3', '#F44336']
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("SHAP Feature Importance")
    try:
        st.image('models/shap_importance.png',
                 use_column_width=True)
    except:
        st.info("Run shap_report.py first to generate this chart")

# ── Predict Customer ─────────────────────
elif page == "Predict Customer":
    st.title("Predict Customer Churn")
    st.write("Fill in customer details to get a churn prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        tenure         = st.slider("Tenure (months)", 0, 72, 12)
        monthly        = st.slider("Monthly Charges ($)", 18, 120, 65)
        total_charges  = st.number_input("Total Charges ($)",
                                          0.0, 9000.0, 800.0)

    with col2:
        senior         = st.selectbox("Senior Citizen", [0, 1])
        partner        = st.selectbox("Partner", [0, 1])
        dependents     = st.selectbox("Dependents", [0, 1])

    with col3:
        paperless      = st.selectbox("Paperless Billing", [0, 1])
        phone_service  = st.selectbox("Phone Service", [0, 1])
        gender         = st.selectbox("Gender", [0, 1],
                                       format_func=lambda x:
                                       "Male" if x==1 else "Female")

    if st.button("Predict Churn", type="primary"):
        # Build input matching training feature columns
        input_dict = {col: 0 for col in df.drop(
                          columns=['Churn']).columns}
        input_dict['tenure']         = tenure
        input_dict['MonthlyCharges'] = monthly
        input_dict['TotalCharges']   = total_charges
        input_dict['SeniorCitizen']  = senior
        input_dict['Partner']        = partner
        input_dict['Dependents']     = dependents
        input_dict['PaperlessBilling'] = paperless
        input_dict['PhoneService']   = phone_service
        input_dict['gender']         = gender

        input_df = pd.DataFrame([input_dict])
        prob     = model.predict_proba(input_df)[0][1]
        pred     = model.predict(input_df)[0]

        st.divider()
        if pred == 1:
            st.error(f"⚠️ High Churn Risk — "
                     f"Probability: {prob*100:.1f}%")
            st.write("**Recommendation:** Offer a discount "
                     "or loyalty plan to retain this customer.")
        else:
            st.success(f"✅ Low Churn Risk — "
                       f"Probability: {prob*100:.1f}%")
            st.write("**Status:** This customer is likely to stay.")

        fig = go.Figure(go.Indicator(
            mode  = "gauge+number",
            value = round(prob * 100, 1),
            title = {"text": "Churn Probability (%)"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar':  {'color': "#F44336" if pred==1
                                  else "#2196F3"},
                'steps': [
                    {'range': [0, 40],   'color': "#E3F2FD"},
                    {'range': [40, 70],  'color': "#FFF9C4"},
                    {'range': [70, 100], 'color': "#FFEBEE"},
                ]
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

# ── SHAP Insights ─────────────────────────
elif page == "SHAP Insights":
    st.title("SHAP Explainability Report")

    st.subheader("Feature Importance")
    try:
        st.image('models/shap_importance.png',
                 use_column_width=True)
    except:
        st.info("Run shap_report.py first")

    st.subheader("Feature Impact Direction")
    try:
        st.image('models/shap_beeswarm.png',
                 use_column_width=True)
    except:
        st.info("Run shap_report.py first")

    st.subheader("Single Prediction Explanation")
    try:
        st.image('models/shap_single_prediction.png',
                 use_column_width=True)
    except:
        st.info("Run shap_report.py first")
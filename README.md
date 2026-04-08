# Customer Churn Prediction Pipeline

## Problem Statement
Predict which telecom customers are likely to churn
using an end-to-end ML pipeline with MLflow tracking.

## Pipeline Architecture
Raw Data → Preprocessing → Feature Store
→ SMOTE Balancing → Model Training (4 models)
→ MLflow Autolog → SHAP Explainability
→ Model Registry (Staging → Production)
→ Streamlit Dashboard

## Tech Stack
- Python, Pandas, Scikit-learn
- XGBoost, LightGBM
- MLflow (tracking + registry)
- SHAP (explainability)
- Streamlit (dashboard)
- Docker (deployment)

## Results
| Model               | Accuracy | F1    | AUC   |
|---------------------|----------|-------|-------|
| Logistic Regression | 0.XX     | 0.XX  | 0.XX  |
| Random Forest       | 0.XX     | 0.XX  | 0.XX  |
| XGBoost             | 0.XX     | 0.XX  | 0.XX  |
| LightGBM            | 0.XX     | 0.XX  | 0.XX  |

## How to Run
# 1. Install dependencies
pip install -r requirements.txt

# 2. Preprocess data
python src/preprocess.py

# 3. Train models
python src/train.py

# 4. Generate SHAP report
python src/shap_report.py

# 5. Register best model
python src/register_model.py

# 6. Launch dashboard
streamlit run app/app.py

# 7. View MLflow UI
mlflow ui

## Docker
docker build -t churn-predictor .
docker run -p 8501:8501 churn-predictor
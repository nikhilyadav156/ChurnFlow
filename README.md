# 🔮 Customer Churn Prediction Pipeline

> End-to-end ML pipeline to predict telecom customer churn — with MLflow tracking,
> SHAP explainability, model registry and a beautiful Streamlit dashboard.

---

## Problem Statement
Predict which telecom customers are likely to churn using a production-grade
ML pipeline built on the IBM Telco Customer Churn dataset (~7,000 customers, 21 features).

---

## Pipeline Architecture

```
Raw Data (IBM Telco)
    ↓
Preprocessing  →  Feature Engineering  →  Feature Store
    ↓
SMOTE Balancing (handles class imbalance ~27% churn)
    ↓
Model Training × 4  ─────────────────────────────────────┐
  • Logistic Regression                                    │
  • Random Forest                                          │  MLflow
  • XGBoost                                                │  Autolog
  • LightGBM                                               │
    ↓                                                      │
5-Fold Stratified Cross-Validation  ←────────────────────┘
    ↓
SHAP Explainability Reports
    ↓
MLflow Model Registry  (Staging → Production)
    ↓
Streamlit Dashboard  ←──  Docker Deployment
```

---

## Tech Stack

| Layer            | Libraries                                        |
|------------------|--------------------------------------------------|
| Data             | Pandas, NumPy                                    |
| ML               | Scikit-learn, XGBoost, LightGBM                  |
| Imbalance        | imbalanced-learn (SMOTE)                         |
| Tracking         | MLflow (SQLite backend)                          |
| Explainability   | SHAP                                             |
| Visualisation    | Plotly, Seaborn, Matplotlib                      |
| Dashboard        | Streamlit                                        |
| Deployment       | Docker                                           |

---

## Results (5-Fold Cross-Validation)

| Model               | CV AUC | CV F1 | CV Accuracy | Precision | Recall |
|---------------------|--------|-------|-------------|-----------|--------|
| Logistic Regression | ~0.84  | ~0.62 | ~0.79       | ~0.66     | ~0.58  |
| Random Forest       | ~0.85  | ~0.62 | ~0.80       | ~0.68     | ~0.57  |
| XGBoost             | ~0.86  | ~0.63 | ~0.81       | ~0.70     | ~0.58  |
| LightGBM            | ~0.87  | ~0.64 | ~0.81       | ~0.71     | ~0.59  |

> Exact values are populated in `artifacts/results.json` after training.

---

## Project Structure

```
.
├── app/
│   └── app.py                  ← Streamlit dashboard (6 pages)
├── artifacts/
│   ├── results.json            ← Training metrics for all models
│   ├── best_model.txt          ← Name of champion model
│   ├── shap/                   ← SHAP plots and CSV values
│   └── cm_*.png                ← Confusion matrices
├── data/
│   ├── raw/                    ← Downloaded CSV (gitignored)
│   └── processed/              ← features.csv, labels.csv
├── models/
│   ├── preprocessor.joblib     ← Fitted ColumnTransformer
│   ├── Logistic_Regression.joblib
│   ├── Random_Forest.joblib
│   ├── XGBoost.joblib
│   └── LightGBM.joblib
├── src/
│   ├── preprocess.py           ← Download, clean, engineer, save
│   ├── train.py                ← SMOTE + MLflow autolog + 4 models
│   ├── shap_report.py          ← SHAP summaries, bar, waterfall
│   └── register_model.py       ← MLflow Model Registry
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## How to Run

```bash
# 1. Clone and install dependencies
pip install -r requirements.txt

# 2. Preprocess data (auto-downloads the Telco dataset)
python src/preprocess.py

# 3. Train all 4 models with MLflow tracking
python src/train.py

# 4. Generate SHAP explainability reports
python src/shap_report.py

# 5. Register the best model to MLflow Model Registry
python src/register_model.py

# 6. Launch the Streamlit dashboard
streamlit run app/app.py

# 7. Open the MLflow UI (separate terminal)
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

---

## Docker

```bash
# Build
docker build -t churn-predictor .

# Run (dashboard on :8501)
docker run -p 8501:8501 churn-predictor
```

---

## Dashboard Pages

| Page                | Description                                                    |
|---------------------|----------------------------------------------------------------|
| 🏠 Overview          | KPI cards, pipeline diagram, quick model comparison            |
| 🔍 Single Prediction | Interactive form → gauge chart + probability score             |
| 📊 Batch Prediction  | Upload CSV → risk tiers + download predictions                 |
| 📈 Model Comparison  | Radar chart, AUC error bars, leaderboard table                 |
| 🧠 SHAP              | Beeswarm, bar chart, waterfall, interactive Plotly importance  |
| 📋 MLflow Runs       | Run explorer, best-run details, link to MLflow UI              |

---

## Key Design Decisions

- **SMOTE** applied only on training data within each CV fold to prevent data leakage.
- **MLflow SQLite** backend — zero-setup, portable, works locally and in Docker.
- **ColumnTransformer** persisted as `preprocessor.joblib` so the dashboard can
  transform raw user input identically to training.
- **Model Registry** archives older Production versions on each promotion to
  keep the registry clean.

---

*Built by Nikhil Yadav .*
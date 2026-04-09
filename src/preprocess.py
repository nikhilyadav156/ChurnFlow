"""
preprocess.py
─────────────
Downloads the IBM Telco Customer-Churn dataset, cleans it,
engineers features, and saves processed artefacts to disk.

Outputs
-------
data/processed/features.csv   – scaled feature matrix
data/processed/labels.csv     – target column (Churn)
data/processed/feature_names.txt
models/preprocessor.joblib    – fitted ColumnTransformer (for inference)
"""

import os, pathlib, requests, warnings
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = pathlib.Path(__file__).resolve().parent.parent
RAW_DIR    = BASE_DIR / "data" / "raw"
PROC_DIR   = BASE_DIR / "data" / "processed"
MODEL_DIR  = BASE_DIR / "models"

for d in [RAW_DIR, PROC_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RAW_CSV  = RAW_DIR  / "telco_churn.csv"
FEAT_CSV = PROC_DIR / "features.csv"
LABL_CSV = PROC_DIR / "labels.csv"
FEAT_TXT = PROC_DIR / "feature_names.txt"
PREP_PKL = MODEL_DIR / "preprocessor.joblib"

# ── Dataset URL ──────────────────────────────────────────────────────────────
DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d"
    "/master/data/Telco-Customer-Churn.csv"
)

# ─────────────────────────────────────────────────────────────────────────────
def download_data() -> pd.DataFrame:
    if RAW_CSV.exists():
        print(f"[preprocess] Raw data already at {RAW_CSV} – skipping download.")
        return pd.read_csv(RAW_CSV)

    print(f"[preprocess] Downloading dataset from {DATA_URL} …")
    r = requests.get(DATA_URL, timeout=30)
    r.raise_for_status()
    RAW_CSV.write_bytes(r.content)
    print(f"[preprocess] Saved to {RAW_CSV}")
    return pd.read_csv(RAW_CSV)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop customer ID – not a feature
    df.drop(columns=["customerID"], inplace=True, errors="ignore")

    # Fix TotalCharges: whitespace strings → NaN → fill with median
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Binary target
    df["Churn"] = (df["Churn"].str.strip().str.lower() == "yes").astype(int)

    # Replace "No internet service" / "No phone service" with "No"
    df.replace(
        {"No internet service": "No", "No phone service": "No"},
        inplace=True
    )

    print(f"[preprocess] Shape after cleaning: {df.shape}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Tenure buckets
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 60, np.inf],
        labels=["0-1yr", "1-2yr", "2-4yr", "4-5yr", "5+yr"]
    ).astype(str)

    # Revenue per month normalised by tenure
    df["revenue_per_month"] = df["TotalCharges"] / (df["tenure"] + 1)

    # High-value flag
    df["high_value"] = (df["MonthlyCharges"] > df["MonthlyCharges"].median()).astype(int)

    # Number of additional services subscribed
    service_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    df["num_services"] = df[service_cols].apply(
        lambda row: (row == "Yes").sum(), axis=1
    )

    print(f"[preprocess] Shape after feature engineering: {df.shape}")
    return df


def build_preprocessor(df: pd.DataFrame, target: str = "Churn"):
    X = df.drop(columns=[target])

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    print(f"[preprocess] Numeric features  : {num_cols}")
    print(f"[preprocess] Categorical feats : {cat_cols}")

    num_pipeline = Pipeline([("scaler", StandardScaler())])
    cat_pipeline = Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols),
    ])

    return preprocessor, num_cols, cat_cols


def preprocess():
    print("\n" + "="*60)
    print("  CUSTOMER CHURN — DATA PREPROCESSING")
    print("="*60 + "\n")

    df = download_data()
    df = clean(df)
    df = engineer_features(df)

    target = "Churn"
    X_raw = df.drop(columns=[target])
    y     = df[target]

    preprocessor, num_cols, cat_cols = build_preprocessor(df, target)

    X_processed = preprocessor.fit_transform(X_raw)

    # Derive feature names after OHE
    ohe_feature_names = (
        preprocessor.named_transformers_["cat"]["ohe"]
        .get_feature_names_out(cat_cols)
        .tolist()
    )
    all_feature_names = num_cols + ohe_feature_names

    X_df = pd.DataFrame(X_processed, columns=all_feature_names)

    # Save artefacts
    X_df.to_csv(FEAT_CSV, index=False)
    y.to_csv(LABL_CSV, index=False)
    FEAT_TXT.write_text("\n".join(all_feature_names))
    joblib.dump(preprocessor, PREP_PKL)

    print(f"\n[preprocess] ✅ Features saved  → {FEAT_CSV}")
    print(f"[preprocess] ✅ Labels saved    → {LABL_CSV}")
    print(f"[preprocess] ✅ Preprocessor    → {PREP_PKL}")
    print(f"[preprocess] ✅ Feature count   : {len(all_feature_names)}")
    print(f"[preprocess] ✅ Churn rate      : {y.mean():.2%}")

    return X_df, y


if __name__ == "__main__":
    preprocess()
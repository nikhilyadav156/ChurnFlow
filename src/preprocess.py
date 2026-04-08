import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

def load_raw_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Fix TotalCharges (has spaces, should be numeric)
    df['TotalCharges'] = pd.to_numeric(
        df['TotalCharges'], errors='coerce'
    )
    # Fill nulls with median
    df['TotalCharges'].fillna(
        df['TotalCharges'].median(), inplace=True
    )
    # Drop customerID
    df = df.drop(columns=['customerID'])
    # Convert target to binary
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    return df

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    binary_cols = [
        'gender', 'Partner', 'Dependents',
        'PhoneService', 'PaperlessBilling'
    ]
    for col in binary_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

    # One-hot encode multi-category columns
    multi_cols = [
        'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies',
        'Contract', 'PaymentMethod'
    ]
    df = pd.get_dummies(df, columns=multi_cols, drop_first=True)
    return df

def scale_features(df: pd.DataFrame,
                   scaler=None,
                   fit: bool = True):
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    if fit:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        joblib.dump(scaler, 'models/scaler.pkl')
    else:
        df[numeric_cols] = scaler.transform(df[numeric_cols])
    return df, scaler

def run_preprocessing(raw_path: str,
                      save_path: str) -> None:
    print("Loading raw data...")
    df = load_raw_data(raw_path)

    print("Cleaning data...")
    df = clean_data(df)

    print("Encoding features...")
    df = encode_features(df)

    print("Scaling features...")
    df, scaler = scale_features(df)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Processed data saved to {save_path}")
    return df

if __name__ == "__main__":
    run_preprocessing(
        raw_path='data/raw/telco_churn.csv',
        save_path='data/processed/features.csv'
    )
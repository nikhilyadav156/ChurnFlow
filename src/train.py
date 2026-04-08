import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score,
                              roc_auc_score, classification_report)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import joblib
import os

PROCESSED_PATH = 'data/processed/features.csv'
MLFLOW_EXPERIMENT = 'customer-churn-prediction'

def load_features(path: str):
    df = pd.read_csv(path)
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    return X, y

def apply_smote(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE — Class distribution: "
          f"{pd.Series(y_resampled).value_counts().to_dict()}")
    return X_resampled, y_resampled

def evaluate_model(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        'accuracy' : round(accuracy_score(y_test, y_pred), 4),
        'f1_score' : round(f1_score(y_test, y_pred), 4),
        'roc_auc'  : round(roc_auc_score(y_test, y_prob), 4),
    }

def train_model(model_name: str,
                model,
                X_train, y_train,
                X_test, y_test) -> dict:

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=model_name):
        # Autolog captures params, metrics, model automatically
        mlflow.autolog()

        print(f"\nTraining {model_name}...")
        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test)

        # Log custom metrics on top of autolog
        mlflow.log_metric('custom_f1',  metrics['f1_score'])
        mlflow.log_metric('custom_auc', metrics['roc_auc'])
        mlflow.log_param('smote_applied', True)
        mlflow.log_param('model_name', model_name)

        # Save model locally
        os.makedirs('models', exist_ok=True)
        joblib.dump(model, f'models/{model_name}.pkl')
        mlflow.log_artifact(f'models/{model_name}.pkl')

        print(f"{model_name} → "
              f"Accuracy: {metrics['accuracy']} | "
              f"F1: {metrics['f1_score']} | "
              f"AUC: {metrics['roc_auc']}")

        return {
            'model_name': model_name,
            'model': model,
            **metrics
        }

def run_training_pipeline():
    print("Loading features...")
    X, y = load_features(PROCESSED_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Applying SMOTE to training data...")
    X_train_bal, y_train_bal = apply_smote(X_train, y_train)

    models = {
        'LogisticRegression': LogisticRegression(
            max_iter=1000, random_state=42
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=100, random_state=42
        ),
        'XGBoost': XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        ),
        'LightGBM': LGBMClassifier(
            random_state=42, verbose=-1
        ),
    }

    results = []
    for name, model in models.items():
        result = train_model(
            name, model,
            X_train_bal, y_train_bal,
            X_test, y_test
        )
        results.append(result)

    # Find best model by AUC
    best = max(results, key=lambda x: x['roc_auc'])
    print(f"\nBest Model: {best['model_name']} "
          f"with AUC = {best['roc_auc']}")

    # Save best model separately
    joblib.dump(best['model'], 'models/best_model.pkl')
    print("Best model saved to models/best_model.pkl")

    return results, best

if __name__ == "__main__":
    results, best = run_training_pipeline()
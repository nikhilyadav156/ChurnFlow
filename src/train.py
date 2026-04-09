"""
train.py
────────
Trains 4 models on the preprocessed Telco Churn dataset with:
  • SMOTE class-balancing
  • MLflow autologging
  • Cross-validation
  • Custom metric logging (AUC-ROC, F1, Precision, Recall)
  • Confusion-matrix artefact
  • Best-model tagging

Run after preprocess.py.
"""

import os, pathlib, warnings, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import joblib

from sklearn.linear_model   import LogisticRegression
from sklearn.ensemble       import RandomForestClassifier
from xgboost                import XGBClassifier
from lightgbm               import LGBMClassifier

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics         import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix,
    classification_report
)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR  = pathlib.Path(__file__).resolve().parent.parent
PROC_DIR  = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
ART_DIR   = BASE_DIR / "artifacts"
ART_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEAT_CSV  = PROC_DIR / "features.csv"
LABL_CSV  = PROC_DIR / "labels.csv"

EXPERIMENT_NAME = "Customer-Churn-Prediction"
MLFLOW_URI      = str(BASE_DIR / "mlflow.db")          # SQLite backend

# ── Model Definitions ────────────────────────────────────────────────────────
MODELS = {
    "Logistic_Regression": LogisticRegression(
        max_iter=1000, C=0.5, class_weight="balanced", random_state=42
    ),
    "Random_Forest": RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=400, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=3, use_label_encoder=False,
        eval_metric="logloss", random_state=42, verbosity=0
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=400, learning_rate=0.05, max_depth=6,
        num_leaves=63, subsample=0.8, colsample_bytree=0.8,
        class_weight="balanced", random_state=42, verbose=-1
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
def load_data():
    X = pd.read_csv(FEAT_CSV)
    y = pd.read_csv(LABL_CSV).squeeze()
    print(f"[train] Loaded  X={X.shape}  y={y.shape}  churn_rate={y.mean():.2%}")
    return X, y


def apply_smote(X_train, y_train):
    sm = SMOTE(random_state=42, k_neighbors=5)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"[train] SMOTE: {X_train.shape[0]} → {X_res.shape[0]} samples")
    return X_res, y_res


def plot_confusion_matrix(cm, model_name: str, save_path: pathlib.Path):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
        ax=ax
    )
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")
    ax.set_ylabel("Actual", fontsize=11)
    ax.set_xlabel("Predicted", fontsize=11)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def train_and_evaluate(X, y):
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_URI}")
    exp = mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"[train] MLflow experiment: '{EXPERIMENT_NAME}'  (id={exp.experiment_id})")

    cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    for model_name, model in MODELS.items():
        print(f"\n{'─'*55}")
        print(f"[train] Training: {model_name}")
        print(f"{'─'*55}")

        # Enable autolog for supported libraries
        if "XGBoost" in model_name:
            mlflow.xgboost.autolog(log_models=True, silent=True)
        elif "LightGBM" in model_name:
            mlflow.lightgbm.autolog(log_models=True, silent=True)
        else:
            mlflow.sklearn.autolog(log_models=True, silent=True)

        with mlflow.start_run(run_name=model_name) as run:
            run_id = run.info.run_id

            # ── Cross-validation ──────────────────────────────────────────
            scoring = {
                "accuracy" : "accuracy",
                "f1"       : "f1",
                "roc_auc"  : "roc_auc",
                "precision": "precision",
                "recall"   : "recall",
            }
            cv_results = cross_validate(
                model, X, y, cv=cv, scoring=scoring,
                return_train_score=True, n_jobs=-1
            )

            # ── Full-data SMOTE fit ───────────────────────────────────────
            X_sm, y_sm = apply_smote(X, y)
            model.fit(X_sm, y_sm)
            y_pred      = model.predict(X)
            y_pred_prob = model.predict_proba(X)[:, 1]

            # ── Metrics ───────────────────────────────────────────────────
            metrics = {
                "cv_accuracy_mean" : float(cv_results["test_accuracy"].mean()),
                "cv_accuracy_std"  : float(cv_results["test_accuracy"].std()),
                "cv_f1_mean"       : float(cv_results["test_f1"].mean()),
                "cv_f1_std"        : float(cv_results["test_f1"].std()),
                "cv_auc_mean"      : float(cv_results["test_roc_auc"].mean()),
                "cv_auc_std"       : float(cv_results["test_roc_auc"].std()),
                "cv_precision_mean": float(cv_results["test_precision"].mean()),
                "cv_recall_mean"   : float(cv_results["test_recall"].mean()),
                "train_accuracy"   : float(accuracy_score(y, y_pred)),
                "train_f1"         : float(f1_score(y, y_pred)),
                "train_auc"        : float(roc_auc_score(y, y_pred_prob)),
            }
            mlflow.log_metrics(metrics)

            # ── Tags ──────────────────────────────────────────────────────
            mlflow.set_tags({
                "model_type"  : model_name,
                "smote"       : "True",
                "cv_folds"    : "5",
                "dataset"     : "IBM-Telco-Churn",
                "developer"   : "nikhil_yadav",
            })

            # ── Confusion matrix artefact ─────────────────────────────────
            cm       = confusion_matrix(y, y_pred)
            cm_path  = ART_DIR / f"cm_{model_name}.png"
            plot_confusion_matrix(cm, model_name, cm_path)
            mlflow.log_artifact(str(cm_path), artifact_path="plots")

            # ── Classification report ─────────────────────────────────────
            report = classification_report(y, y_pred, target_names=["No Churn", "Churn"])
            report_path = ART_DIR / f"report_{model_name}.txt"
            report_path.write_text(report)
            mlflow.log_artifact(str(report_path), artifact_path="reports")

            # ── Save model locally ────────────────────────────────────────
            model_path = MODEL_DIR / f"{model_name}.joblib"
            joblib.dump(model, model_path)
            mlflow.log_artifact(str(model_path), artifact_path="model_files")

            results[model_name] = {
                "run_id" : run_id,
                **metrics
            }

            print(f"  AUC   : {metrics['cv_auc_mean']:.4f} ± {metrics['cv_auc_std']:.4f}")
            print(f"  F1    : {metrics['cv_f1_mean']:.4f} ± {metrics['cv_f1_std']:.4f}")
            print(f"  Acc   : {metrics['cv_accuracy_mean']:.4f}")
            print(f"  RunID : {run_id[:8]}…")

    # ── Summary ───────────────────────────────────────────────────────────────
    summary_df = pd.DataFrame(results).T
    summary_df.to_csv(ART_DIR / "results_summary.csv")

    best_model = max(results, key=lambda k: results[k]["cv_auc_mean"])
    print(f"\n{'='*55}")
    print(f"  🏆 Best Model by CV-AUC: {best_model}")
    print(f"     AUC  = {results[best_model]['cv_auc_mean']:.4f}")
    print(f"     F1   = {results[best_model]['cv_f1_mean']:.4f}")
    print(f"{'='*55}\n")

    # Save best model name for other scripts
    (BASE_DIR / "artifacts" / "best_model.txt").write_text(best_model)
    # Save full results as JSON for dashboard
    with open(ART_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results, best_model


if __name__ == "__main__":
    print("\n" + "="*55)
    print("  CUSTOMER CHURN — MODEL TRAINING")
    print("="*55)

    X, y = load_data()
    results, best_model = train_and_evaluate(X, y)

    print("✅  Training complete. Run  `mlflow ui`  to explore runs.")
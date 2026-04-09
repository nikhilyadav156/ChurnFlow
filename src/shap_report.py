"""
shap_report.py
──────────────
Generates SHAP explainability artefacts for all trained models and
logs them back to the corresponding MLflow runs.

Outputs (in artifacts/shap/)
───────────────────────────
  shap_summary_{model}.png       – Beeswarm summary plot
  shap_bar_{model}.png           – Global mean |SHAP| bar plot
  shap_waterfall_{model}.png     – Single-prediction waterfall
  shap_values_{model}.csv        – Raw SHAP value matrix
"""

import pathlib, warnings, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import joblib
import mlflow

warnings.filterwarnings("ignore")
shap.initjs()

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR  = pathlib.Path(__file__).resolve().parent.parent
PROC_DIR  = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
ART_DIR   = BASE_DIR / "artifacts"
SHAP_DIR  = ART_DIR / "shap"
SHAP_DIR.mkdir(parents=True, exist_ok=True)

FEAT_CSV  = PROC_DIR / "features.csv"
LABL_CSV  = PROC_DIR / "labels.csv"
BEST_TXT  = ART_DIR / "best_model.txt"
RES_JSON  = ART_DIR / "results.json"

EXPERIMENT_NAME = "Customer-Churn-Prediction"
MLFLOW_URI      = str(BASE_DIR / "mlflow.db")

# ── SHAP sample size (keep explainer fast) ───────────────────────────────────
SAMPLE_N = 500

# ─────────────────────────────────────────────────────────────────────────────
def load_data():
    X = pd.read_csv(FEAT_CSV)
    y = pd.read_csv(LABL_CSV).squeeze()
    feat_names = X.columns.tolist()
    return X.values, y.values, feat_names


def get_explainer(model, model_name: str, X_bg):
    """Return the right SHAP explainer for each model type."""
    name = model_name.lower()
    if "logistic" in name:
        return shap.LinearExplainer(model, X_bg, feature_perturbation="correlation_dependent")
    elif "forest" in name:
        return shap.TreeExplainer(model)
    elif "xgboost" in name or "lightgbm" in name:
        return shap.TreeExplainer(model)
    else:
        return shap.KernelExplainer(model.predict_proba, shap.sample(X_bg, 100))


def plot_summary(shap_values, X_sample, feat_names, model_name: str, save_path: pathlib.Path):
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_sample,
        feature_names=np.array(feat_names),
        show=False, plot_size=(10, 8)
    )
    plt.title(f"SHAP Summary — {model_name}", fontsize=14, fontweight="bold", pad=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_bar(shap_values, feat_names, model_name: str, save_path: pathlib.Path):
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values, feature_names=np.array(feat_names),
        plot_type="bar", show=False
    )
    plt.title(f"SHAP Feature Importance (mean |φ|) — {model_name}",
              fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_waterfall(explainer, X_sample, feat_names, model_name: str, save_path: pathlib.Path, idx: int = 0):
    """Single-prediction waterfall for a churned customer (if found)."""
    try:
        sv = explainer(X_sample)

        # Extract values for the single sample
        vals = sv.values[idx]

        # If shape is (features, classes) — tree multi-output → take class 1 (churn)
        if vals.ndim == 2:
            vals = vals[:, 1]

        # base_values can be scalar, 1D, or 2D
        base = sv.base_values[idx] if hasattr(sv.base_values, "__len__") else sv.base_values
        if hasattr(base, "__len__"):
            base = base[1] if len(base) == 2 else base[0]

        exp_obj = shap.Explanation(
            values=vals,
            base_values=float(base),
            data=X_sample[idx],
            feature_names=np.array(feat_names)
        )
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(exp_obj, max_display=15, show=False)
        plt.title(f"SHAP Waterfall — {model_name} (Sample #{idx})",
                  fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [shap] Waterfall saved for {model_name} ✅")
    except Exception as e:
        print(f"  [shap] Waterfall skipped for {model_name}: {e}")


def generate_shap(model_name: str, run_id: str, X, y, feat_names):
    print(f"\n[shap] Processing: {model_name}")

    model_path = MODEL_DIR / f"{model_name}.joblib"
    if not model_path.exists():
        print(f"  [shap] Model not found: {model_path}  — skipping.")
        return

    model   = joblib.load(model_path)
    rng     = np.random.default_rng(42)
    idx_all = rng.choice(len(X), size=min(SAMPLE_N, len(X)), replace=False)
    X_samp  = X[idx_all]

    explainer   = get_explainer(model, model_name, X_samp)

   # Get SHAP values (handle multi-output — take class=1 slice)
    raw_sv = explainer.shap_values(X_samp)
    if isinstance(raw_sv, list):
        # older SHAP: list of arrays, one per class
        sv_churn = np.array(raw_sv[1])
    elif hasattr(raw_sv, "values"):
        sv_churn = raw_sv.values
        if sv_churn.ndim == 3:
            sv_churn = sv_churn[:, :, 1]
    else:
        sv_churn = np.array(raw_sv)
    
    # Final safety — always ensure 2D (samples × features)
    if sv_churn.ndim == 3:
        sv_churn = sv_churn[:, :, 1]
    elif sv_churn.ndim == 1:
        sv_churn = sv_churn.reshape(1, -1)

    # ── Plots ─────────────────────────────────────────────────────────────
    summary_path   = SHAP_DIR / f"shap_summary_{model_name}.png"
    bar_path       = SHAP_DIR / f"shap_bar_{model_name}.png"
    waterfall_path = SHAP_DIR / f"shap_waterfall_{model_name}.png"
    csv_path       = SHAP_DIR / f"shap_values_{model_name}.csv"

    plot_summary(sv_churn, X_samp, feat_names, model_name, summary_path)
    plot_bar(sv_churn, feat_names, model_name, bar_path)
    plot_waterfall(explainer, X_samp, feat_names, model_name, waterfall_path)

    # Save raw SHAP values
    pd.DataFrame(sv_churn, columns=feat_names).to_csv(csv_path, index=False)

    # ── Top features ──────────────────────────────────────────────────────
    mean_abs = np.abs(sv_churn).mean(axis=0)
    top10 = pd.Series(mean_abs, index=feat_names).nlargest(10)
    print("  Top-10 features by mean |SHAP|:")
    for feat, val in top10.items():
        print(f"    {feat:<40} {val:.4f}")

    # ── Log artefacts back to MLflow run ──────────────────────────────────
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_URI}")
    with mlflow.start_run(run_id=run_id):
        for p in [summary_path, bar_path, waterfall_path, csv_path]:
            if p.exists():
                mlflow.log_artifact(str(p), artifact_path="shap")

    print(f"  [shap] ✅ Artefacts logged to MLflow run {run_id[:8]}…")


def shap_report():
    print("\n" + "="*55)
    print("  CUSTOMER CHURN — SHAP EXPLAINABILITY")
    print("="*55)

    X, y, feat_names = load_data()

    if not RES_JSON.exists():
        print("[shap] results.json not found. Run train.py first.")
        return

    with open(RES_JSON) as f:
        results = json.load(f)

    for model_name, info in results.items():
        generate_shap(model_name, info["run_id"], X, y, feat_names)

    print(f"\n✅  SHAP reports saved to {SHAP_DIR}")


if __name__ == "__main__":
    shap_report()
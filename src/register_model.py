"""
register_model.py
─────────────────
Reads the results.json produced by train.py, identifies the best model
by CV-AUC, and registers it in the MLflow Model Registry with the path
Staging → Production.

Usage
-----
  python src/register_model.py                 # auto-picks best by AUC
  python src/register_model.py --model XGBoost # override model choice
"""

import argparse, json, pathlib, time
import mlflow
from mlflow.tracking  import MlflowClient
import joblib

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = pathlib.Path(__file__).resolve().parent.parent
ART_DIR    = BASE_DIR / "artifacts"
MODEL_DIR  = BASE_DIR / "models"
RES_JSON   = ART_DIR / "results.json"
BEST_TXT   = ART_DIR / "best_model.txt"

EXPERIMENT_NAME = "Customer-Churn-Prediction"
MLFLOW_URI      = str(BASE_DIR / "mlflow.db")
REGISTRY_NAME   = "ChurnPredictor"

# ─────────────────────────────────────────────────────────────────────────────
def load_results() -> dict:
    if not RES_JSON.exists():
        raise FileNotFoundError(f"results.json not found at {RES_JSON}. Run train.py first.")
    with open(RES_JSON) as f:
        return json.load(f)


def pick_best_model(results: dict, override: str | None = None) -> tuple[str, str]:
    if override:
        # Normalise: accept 'xgboost' → 'XGBoost' etc.
        matches = [k for k in results if k.lower() == override.lower()]
        if not matches:
            raise ValueError(f"Model '{override}' not found. Available: {list(results.keys())}")
        model_name = matches[0]
    else:
        model_name = max(results, key=lambda k: results[k]["cv_auc_mean"])

    run_id = results[model_name]["run_id"]
    return model_name, run_id


def register_and_promote(model_name: str, run_id: str):
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_URI}")
    client = MlflowClient()

    print(f"\n[register] Model  : {model_name}")
    print(f"[register] Run ID : {run_id}")

    # ── Log model artifact to the run (if not already) ───────────────────
    model_path = MODEL_DIR / f"{model_name}.joblib"
    model_obj  = joblib.load(model_path)

    with mlflow.start_run(run_id=run_id):
        model_uri = mlflow.sklearn.log_model(
            sk_model    = model_obj,
            artifact_path = "production_model",
            registered_model_name = REGISTRY_NAME,
        ).model_uri

    print(f"[register] Logged model → {model_uri}")

    # ── Wait for model version to be registered ──────────────────────────
    time.sleep(3)
    versions = client.search_model_versions(f"name='{REGISTRY_NAME}'")
    if not versions:
        raise RuntimeError("Model version not found in registry after registration.")

    # Pick the version tied to this run
    version_obj = next(
        (v for v in versions if v.run_id == run_id),
        versions[-1]   # fallback to latest
    )
    version = version_obj.version
    print(f"[register] Registered version: {version}")

    # ── Staging → Production ─────────────────────────────────────────────
    client.transition_model_version_stage(
        name    = REGISTRY_NAME,
        version = version,
        stage   = "Staging",
        archive_existing_versions=False,
    )
    print(f"[register] Version {version} → Staging ✅")
    time.sleep(1)

    client.transition_model_version_stage(
        name    = REGISTRY_NAME,
        version = version,
        stage   = "Production",
        archive_existing_versions=True,   # archive older Production versions
    )
    print(f"[register] Version {version} → Production ✅")

    # ── Add model description and tags ───────────────────────────────────
    client.update_model_version(
        name        = REGISTRY_NAME,
        version     = version,
        description = (
            f"Best churn prediction model ({model_name}) — "
            f"promoted automatically by register_model.py. "
            f"CV-AUC and F1 scores logged in MLflow experiment."
        ),
    )
    client.set_model_version_tag(REGISTRY_NAME, version, "promoted_by", "nikhil_yadav")
    client.set_model_version_tag(REGISTRY_NAME, version, "algorithm",   model_name)

    # ── Update best_model.txt ─────────────────────────────────────────────
    BEST_TXT.write_text(model_name)
    (ART_DIR / "production_run_id.txt").write_text(run_id)

    print(f"\n{'='*55}")
    print(f"  🚀  {model_name}  v{version}  is now in Production!")
    print(f"{'='*55}")
    print(f"\n  Registry name : {REGISTRY_NAME}")
    print(f"  Model URI     : models:/{REGISTRY_NAME}/{version}")
    print(f"  Stage         : Production")
    print(f"\n  Run  `mlflow ui`  →  Models tab  to verify.")


def register_model(override: str | None = None):
    print("\n" + "="*55)
    print("  CUSTOMER CHURN — MODEL REGISTRY")
    print("="*55)

    results    = load_results()
    model_name, run_id = pick_best_model(results, override)

    print(f"\n  Available models:")
    for m, r in results.items():
        marker = "  ◀ BEST" if m == model_name and not override else \
                 "  ◀ OVERRIDE" if m == model_name else ""
        print(f"    {m:<30} AUC={r['cv_auc_mean']:.4f}  F1={r['cv_f1_mean']:.4f}{marker}")

    register_and_promote(model_name, run_id)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register best model to MLflow Production.")
    parser.add_argument("--model", type=str, default=None,
                        help="Override model name (e.g. XGBoost, LightGBM)")
    args = parser.parse_args()
    register_model(override=args.model)
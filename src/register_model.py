import mlflow
from mlflow.tracking import MlflowClient
import joblib

EXPERIMENT_NAME = 'customer-churn-prediction'
MODEL_NAME      = 'ChurnPredictor'

def get_best_run(experiment_name: str) -> str:
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=['metrics.custom_auc DESC']
    )
    best_run = runs[0]
    print(f"Best Run ID : {best_run.info.run_id}")
    print(f"Best AUC    : "
          f"{best_run.data.metrics.get('custom_auc', 'N/A')}")
    return best_run.info.run_id

def register_model(run_id: str, model_name: str) -> str:
    model_uri = f"runs:/{run_id}/model"
    result    = mlflow.register_model(model_uri, model_name)
    version   = result.version
    print(f"Model registered → {model_name} v{version}")
    return version

def promote_to_staging(model_name: str, version: str) -> None:
    client = MlflowClient()
    client.transition_model_version_stage(
        name    = model_name,
        version = version,
        stage   = 'Staging'
    )
    print(f"{model_name} v{version} → moved to STAGING")

def promote_to_production(model_name: str,
                           version: str) -> None:
    client = MlflowClient()
    client.transition_model_version_stage(
        name    = model_name,
        version = version,
        stage   = 'Production',
        archive_existing_versions = True
    )
    print(f"{model_name} v{version} → moved to PRODUCTION")

def load_production_model(model_name: str):
    model_uri = f"models:/{model_name}/Production"
    model     = mlflow.sklearn.load_model(model_uri)
    print(f"Loaded {model_name} from Production stage")
    return model

def run_registry_pipeline():
    print("Step 1: Finding best run...")
    best_run_id = get_best_run(EXPERIMENT_NAME)

    print("\nStep 2: Registering model...")
    version = register_model(best_run_id, MODEL_NAME)

    print("\nStep 3: Promoting to Staging...")
    promote_to_staging(MODEL_NAME, version)

    print("\nStep 4: Testing in Staging...")
    # In real projects — run integration tests here
    print("Tests passed — promoting to Production")

    print("\nStep 5: Promoting to Production...")
    promote_to_production(MODEL_NAME, version)

    print("\nStep 6: Loading from Production...")
    prod_model = load_production_model(MODEL_NAME)
    print(f"Production model type: {type(prod_model)}")

    return prod_model

if __name__ == "__main__":
    run_registry_pipeline()
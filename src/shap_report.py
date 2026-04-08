import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import mlflow
import warnings
warnings.filterwarnings('ignore')

def generate_shap_report(model_path: str,
                          data_path: str,
                          experiment_name: str) -> None:

    print("Loading model and data...")
    model = joblib.load(model_path)
    df    = pd.read_csv(data_path)
    X     = df.drop(columns=['Churn'])
    y     = df['Churn']

    X_sample = X.sample(n=500, random_state=42)

    print("Computing SHAP values...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name='SHAP_Explainability'):

        # Plot 1 — Summary plot (feature importance)
        plt.figure(figsize=(10, 7))
        shap.summary_plot(shap_values, X_sample,
                          plot_type='bar', show=False)
        plt.title('SHAP Feature Importance')
        plt.tight_layout()
        plt.savefig('models/shap_importance.png',
                    bbox_inches='tight', dpi=150)
        mlflow.log_artifact('models/shap_importance.png')
        plt.close()
        print("SHAP importance plot saved.")

        # Plot 2 — Beeswarm plot (impact direction)
        plt.figure(figsize=(10, 7))
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.title('SHAP Beeswarm — Feature Impact Direction')
        plt.tight_layout()
        plt.savefig('models/shap_beeswarm.png',
                    bbox_inches='tight', dpi=150)
        mlflow.log_artifact('models/shap_beeswarm.png')
        plt.close()
        print("SHAP beeswarm plot saved.")

        # Plot 3 — Single prediction explanation
        plt.figure(figsize=(12, 4))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=X_sample.iloc[0],
                feature_names=X_sample.columns.tolist()
            ), show=False
        )
        plt.tight_layout()
        plt.savefig('models/shap_single_prediction.png',
                    bbox_inches='tight', dpi=150)
        mlflow.log_artifact('models/shap_single_prediction.png')
        plt.close()
        print("SHAP single prediction plot saved.")

        # Log top 5 features
        shap_df = pd.DataFrame({
            'feature': X_sample.columns,
            'importance': np.abs(shap_values).mean(axis=0)
        }).sort_values('importance', ascending=False)

        top5 = shap_df.head(5)
        for i, row in top5.iterrows():
            mlflow.log_metric(
                f'shap_importance_{row["feature"][:20]}',
                round(row['importance'], 4)
            )

        print("\nTop 5 Features by SHAP:")
        print(top5.to_string(index=False))

if __name__ == "__main__":
    generate_shap_report(
        model_path='models/best_model.pkl',
        data_path='data/processed/features.csv',
        experiment_name='customer-churn-prediction'
    )
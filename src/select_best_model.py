import mlflow
from mlflow.tracking import MlflowClient
import mlflow.sklearn
import os


# ==========================================
# MLFLOW CONNECTION
# ==========================================

mlflow.set_tracking_uri(
    "sqlite:///C:/loan%20default%20mlops/src/mlflow.db"
)

client = MlflowClient()


# ==========================================
# EXPERIMENT
# ==========================================

experiment = client.get_experiment_by_name(
    "Loan_Default_Classification"
)

if experiment is None:
    print("Experiment not found.")
    exit()


print("Experiment found:")
print(experiment.name)


# ==========================================
# GET SUCCESSFUL RUNS
# ==========================================

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="attributes.status = 'FINISHED'",
    order_by=["metrics.roc_auc DESC"]
)


if not runs:
    print("No successful runs found.")
    exit()


# ==========================================
# DISPLAY MODEL RESULTS
# ==========================================

print("\nModel Results:")
print("------------------------------------------")

for run in runs:

    model_name = run.data.params.get(
        "model",
        "Unknown"
    )

    roc_auc = run.data.metrics.get(
        "roc_auc",
        0
    )

    accuracy = run.data.metrics.get(
        "accuracy",
        0
    )

    f1 = run.data.metrics.get(
        "f1_score",
        0
    )

    print(
        f"{model_name} | "
        f"Accuracy: {accuracy:.4f} | "
        f"F1: {f1:.4f} | "
        f"ROC-AUC: {roc_auc:.4f}"
    )


# ==========================================
# SELECT BEST MODEL
# ==========================================

best_run = runs[0]

best_model_name = best_run.data.params.get(
    "model"
)

best_roc_auc = best_run.data.metrics.get(
    "roc_auc"
)

best_run_id = best_run.info.run_id


print("\n==========================================")
print("BEST MODEL")
print("==========================================")

print("Model:", best_model_name)
print("ROC-AUC:", round(best_roc_auc, 4))
print("Run ID:", best_run_id)


# ==========================================
# GET MODEL FROM MLFLOW
# ==========================================

model_uri = f"runs:/{best_run_id}/model"

print("\nLoading model from MLflow...")

best_model = mlflow.sklearn.load_model(
    model_uri
)


# ==========================================
# CREATE MODELS FOLDER
# ==========================================

os.makedirs(
    "../models",
    exist_ok=True
)


# ==========================================
# SAVE BEST MODEL
# ==========================================

model_path = "../models/best_model.pkl"

import joblib

joblib.dump(
    best_model,
    model_path
)


print("\nBest model saved successfully.")

print("Saved at:", model_path)

print("\n==========================================")
print("MODEL SELECTION COMPLETED")
print("==========================================")
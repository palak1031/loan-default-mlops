import mlflow

# MLflow tracking database
MLFLOW_URI = "sqlite:///C:/loan%20default%20mlops/src/mlflow.db"

# Best model run ID
RUN_ID = "43056e5f4cc44b47b041728550bc5ed3"

# Registered model name
MODEL_NAME = "LoanDefaultModel"

# Connect to MLflow
mlflow.set_tracking_uri(MLFLOW_URI)

# Model location inside the MLflow run
model_uri = f"runs:/{RUN_ID}/model"

print("\nRegistering best model in MLflow...")

# Register model
result = mlflow.register_model(
    model_uri=model_uri,
    name=MODEL_NAME
)

print("\n========== MODEL REGISTERED ==========")
print("Model Name :", result.name)
print("Model Version :", result.version)
print("Run ID :", RUN_ID)
print("======================================\n")
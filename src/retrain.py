import os
import shutil
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# ==============================
# FILE PATHS
# ==============================

DATA_PATH = "data/loan_df_train.csv"
MODEL_PATH = "models/best_model.pkl"
MONITORING_PATH = "monitoring/monitoring_result.csv"

# MLflow database used by the project
MLFLOW_URI = "sqlite:///C:/loan%20default%20mlops/src/mlflow.db"

# Performance thresholds
MIN_F1 = 0.70
MIN_ROC_AUC = 0.90


print("\n========== AUTOMATIC RETRAINING ==========\n")


# ==============================
# CHECK CURRENT PERFORMANCE
# ==============================

monitoring = pd.read_csv(MONITORING_PATH)

current_f1 = float(monitoring["f1_score"].iloc[-1])
current_roc_auc = float(monitoring["roc_auc"].iloc[-1])

print(f"Current F1 Score : {current_f1:.4f}")
print(f"Current ROC-AUC  : {current_roc_auc:.4f}")

if current_f1 >= MIN_F1 and current_roc_auc >= MIN_ROC_AUC:
    print("\n✓ Current model performance is acceptable.")
    print("No retraining is required at this time.")
    print("\n========== RETRAINING CHECK COMPLETED ==========\n")
    exit()


# ==============================
# LOAD DATA
# ==============================

print("\n⚠ Model performance is below the threshold.")
print("Starting automatic retraining...\n")

df = pd.read_csv(DATA_PATH)

if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

X = df.drop(columns=["Status"])
y = df["Status"]


# ==============================
# TRAIN / TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==============================
# PREPROCESSING
# ==============================

categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    exclude=["object"]
).columns.tolist()


numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numerical_columns),
    ("categorical", categorical_pipeline, categorical_columns)
])


# ==============================
# NEW RANDOM FOREST MODEL
# ==============================

new_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)


pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", new_model)
])


print("Training new Random Forest model...")

pipeline.fit(X_train, y_train)


# ==============================
# EVALUATE NEW MODEL
# ==============================

predictions = pipeline.predict(X_test)
probabilities = pipeline.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
roc_auc = roc_auc_score(y_test, probabilities)


print("\n---------- NEW MODEL PERFORMANCE ----------")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")


# ==============================
# LOG NEW MODEL IN MLFLOW
# ==============================

mlflow.set_tracking_uri(MLFLOW_URI)

experiment_name = "Loan_Default_Classification"

experiment = mlflow.get_experiment_by_name(experiment_name)

if experiment is None:
    experiment_id = mlflow.create_experiment(experiment_name)
else:
    experiment_id = experiment.experiment_id


with mlflow.start_run(
    experiment_id=experiment_id,
    run_name="Automatic_Retraining"
):

    mlflow.log_param("model", "Random Forest")
    mlflow.log_param("retraining", "automatic")

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)

    mlflow.sklearn.log_model(
        pipeline,
        name="model",
        skops_trusted_types=[
            "numpy.dtype",
            "sklearn.tree._tree.Tree"
        ]
    )

    run_id = mlflow.active_run().info.run_id


print("\nNew model logged in MLflow.")
print("Run ID:", run_id)


# ==============================
# MODEL COMPARISON
# ==============================

current_model = joblib.load(MODEL_PATH)

current_predictions = current_model.predict(X_test)
current_probabilities = current_model.predict_proba(X_test)[:, 1]

current_model_f1 = f1_score(
    y_test,
    current_predictions
)

current_model_roc_auc = roc_auc_score(
    y_test,
    current_probabilities
)


print("\n---------- MODEL COMPARISON ----------")
print(f"Current Model ROC-AUC : {current_model_roc_auc:.4f}")
print(f"New Model ROC-AUC     : {roc_auc:.4f}")


# ==============================
# PROMOTE BETTER MODEL
# ==============================

if roc_auc > current_model_roc_auc:

    versioned_path = "models/best_model_retrained.pkl"

    joblib.dump(
        pipeline,
        versioned_path
    )

    shutil.copy(
        versioned_path,
        MODEL_PATH
    )

    print("\n✓ New model performs better.")
    print("✓ New model promoted as the best model.")
    print("✓ Model file updated:", MODEL_PATH)

else:

    print("\n✓ Current model is still better.")
    print("✓ Existing model remains deployed.")


print("\n========== AUTOMATIC RETRAINING COMPLETED ==========\n")
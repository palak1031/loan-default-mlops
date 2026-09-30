import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# File paths
DATA_PATH = "data/loan_df_train.csv"
MODEL_PATH = "models/best_model.pkl"

# Performance thresholds
MIN_F1 = 0.70
MIN_ROC_AUC = 0.90

print("\n========== MODEL MONITORING ==========\n")

# Load dataset
df = pd.read_csv(DATA_PATH)

# Remove unnecessary column
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# Separate features and target
X = df.drop(columns=["Status"])
y = df["Status"]

# Recreate the same test split used during training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Load deployed model
model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")
print("Monitoring test samples:", len(X_test))

# Make predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

# Calculate performance metrics
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
roc_auc = roc_auc_score(y_test, probabilities)

print("\n---------- MODEL PERFORMANCE ----------")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")

# Check whether retraining is required
if f1 < MIN_F1 or roc_auc < MIN_ROC_AUC:
    retraining_required = True
    print("\n⚠️ Retraining required.")
else:
    retraining_required = False
    print("\n✓ Model performance is acceptable.")

# Save monitoring result
monitoring_result = pd.DataFrame([{
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "roc_auc": roc_auc,
    "retraining_required": retraining_required
}])

monitoring_result.to_csv(
    "monitoring_result.csv",
    index=False
)

print("\nMonitoring result saved to:")
print("monitoring/monitoring_result.csv")

print("\n========== MONITORING COMPLETED ==========\n")
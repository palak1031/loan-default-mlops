import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# =========================================================
# LOAD DATASET
# =========================================================

DATA_PATH = "../data/loan_df_train.csv"

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully")
print("Dataset shape:", df.shape)


# =========================================================
# REMOVE UNNECESSARY COLUMN
# =========================================================

if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

print("\nRemoved unnecessary index column if present.")


# =========================================================
# TARGET AND FEATURES
# =========================================================

X = df.drop(columns=["Status"])
y = df["Status"]

print("\nTarget column: Status")

print("\nTarget distribution:")
print(y.value_counts())


# =========================================================
# IDENTIFY CATEGORICAL AND NUMERICAL COLUMNS
# =========================================================

categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    exclude=["object"]
).columns.tolist()

print("\nCategorical columns:")
print(categorical_columns)

print("\nNumerical columns:")
print(numerical_columns)


# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# =========================================================
# NUMERICAL PREPROCESSING
# =========================================================

numerical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)


# =========================================================
# CATEGORICAL PREPROCESSING
# =========================================================

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)


# =========================================================
# COMBINE PREPROCESSING
# =========================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            numerical_pipeline,
            numerical_columns
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_columns
        )
    ]
)


# =========================================================
# MODELS
# =========================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        random_state=42
    )
}


# =========================================================
# MLFLOW EXPERIMENT
# =========================================================

mlflow.set_experiment("Loan_Default_Classification")


# =========================================================
# TRAIN EACH MODEL
# =========================================================

for model_name, model in models.items():

    print("\n========================================")
    print("Training:", model_name)
    print("========================================")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    with mlflow.start_run(run_name=model_name):

        # -------------------------------------------------
        # TRAIN
        # -------------------------------------------------

        pipeline.fit(X_train, y_train)

        # -------------------------------------------------
        # PREDICTION
        # -------------------------------------------------

        y_pred = pipeline.predict(X_test)

        y_probability = pipeline.predict_proba(X_test)[:, 1]

        # -------------------------------------------------
        # METRICS
        # -------------------------------------------------

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        roc_auc = roc_auc_score(
            y_test,
            y_probability
        )

        # -------------------------------------------------
        # DISPLAY RESULTS
        # -------------------------------------------------

        print("\nAccuracy :", round(accuracy, 4))
        print("Precision:", round(precision, 4))
        print("Recall   :", round(recall, 4))
        print("F1 Score :", round(f1, 4))
        print("ROC-AUC  :", round(roc_auc, 4))

        # -------------------------------------------------
        # MLFLOW PARAMETERS
        # -------------------------------------------------

        mlflow.log_param(
            "model",
            model_name
        )

        mlflow.log_param(
            "test_size",
            0.20
        )

        mlflow.log_param(
            "random_state",
            42
        )

        # -------------------------------------------------
        # MLFLOW METRICS
        # -------------------------------------------------

        mlflow.log_metric(
            "accuracy",
            accuracy
        )

        mlflow.log_metric(
            "precision",
            precision
        )

        mlflow.log_metric(
            "recall",
            recall
        )

        mlflow.log_metric(
            "f1_score",
            f1
        )

        mlflow.log_metric(
            "roc_auc",
            roc_auc
        )

        # -------------------------------------------------
        # SAVE MODEL TO MLFLOW
        # -------------------------------------------------

        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            skops_trusted_types=[
                "numpy.dtype",
                "sklearn.tree._tree.Tree"
            ]
        )

        print("\nMLflow logging completed for:", model_name)


# =========================================================
# COMPLETION MESSAGE
# =========================================================

print("\n")
print("========================================")
print("ALL MODELS TRAINED SUCCESSFULLY")
print("MLFLOW EXPERIMENTS LOGGED SUCCESSFULLY")
print("========================================")
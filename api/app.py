from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import joblib


# ==========================================
# CREATE FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="Loan Default Prediction API",
    description="API for predicting loan default risk",
    version="1.0"
)


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

MODEL_PATH = "../models/best_model.pkl"

model = joblib.load(MODEL_PATH)

print("Best model loaded successfully")


# ==========================================
# INPUT DATA MODEL
# ==========================================

class LoanData(BaseModel):

    loan_limit: str
    Gender: str

    approv_in_adv: float
    loan_type: str
    loan_purpose: str
    Credit_Worthiness: str

    open_credit: float
    business_or_commercial: float

    term: float

    Neg_ammortization: float
    interest_only: float
    lump_sum_payment: float

    occupancy_type: str

    total_units: float

    credit_type: str

    co_applicant_credit_type: str = Field(
        alias="co.applicant_credit_type"
    )

    age: str
    submission_of_application: str
    Region: str

    PC1: float
    PC2: float
    PC3: float
    PC4: float
    PC5: float


# ==========================================
# HOME ENDPOINT
# ==========================================

@app.get("/")
def home():

    return {
        "message": "Loan Default Prediction API is running"
    }


# ==========================================
# HEALTH CHECK
# ==========================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model": "Random Forest"
    }


# ==========================================
# PREDICTION ENDPOINT
# ==========================================

@app.post("/predict")
def predict(data: LoanData):

    # Convert input to dictionary
    input_data = data.model_dump(
        by_alias=True
    )

    # Convert input into DataFrame
    input_df = pd.DataFrame([input_data])

    # Make prediction
    prediction = model.predict(input_df)[0]

    # Get probability of default
    probability = model.predict_proba(
        input_df
    )[0][1]

    # Convert prediction to readable result
    if prediction == 1:
        result = "Default"
    else:
        result = "No Default"

    return {
        "prediction": int(prediction),
        "result": result,
        "default_probability": round(
            float(probability),
            4
        )
    }


# ==========================================
# API READY MESSAGE
# ==========================================

print("Loan Default Prediction API ready")
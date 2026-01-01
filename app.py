from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI(
    title="Loan Approval Prediction API",
    description="Predict loan approval using Logistic Regression and Decision Tree models",
    version="1.0"
)

# Load models and preprocessing objects
log_model = joblib.load("models/logistic_model.joblib")
dt_model = joblib.load("models/decision_tree_model.joblib")
scaler = joblib.load("models/scaler.joblib")
feature_columns = joblib.load("models/feature_columns.joblib")

# Input SChema
class LoanApplication(BaseModel):
    no_of_dependents: int
    education: str
    self_employed: str
    income_annum: float
    loan_amount: float
    loan_term: int
    cibil_score: int
    residential_assets_value: float
    commercial_assets_value: float
    luxury_assets_value: float
    bank_asset_value: float
    
# Process Input Data

def preprocess_input(data: LoanApplication):
    # Convert input to DataFrame
    input_df = pd.DataFrame([data.dict()])

    # One-hot encode
    input_df = pd.get_dummies(input_df)

    # Add missing columns (if any)
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Ensure correct column order
    input_df = input_df[feature_columns]

    return input_df

# Prediction endpoint
@app.post("/predict")
def predict_loan(data: LoanApplication):
    processed_data = preprocess_input(data)

    # Logistic Regression prediction
    scaled_data = scaler.transform(processed_data)
    log_pred = log_model.predict(scaled_data)[0]

    # Decision Tree prediction
    dt_pred = dt_model.predict(processed_data)[0]

    return {
        "logistic_regression": "Approved" if log_pred == 1 else "Rejected",
        "decision_tree": "Approved" if dt_pred == 1 else "Rejected"
    }

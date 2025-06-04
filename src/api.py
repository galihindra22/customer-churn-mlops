# src/api.py
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler
from starlette.responses import JSONResponse

app = FastAPI()

# Load model, encoder, scaler
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")
model_input_columns = joblib.load("models/columns.pkl")



class ChurnInput(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.post("/predict")
def predict(input: ChurnInput):
    # Convert input to DataFrame
    input_df = pd.DataFrame([input.dict()])

    # Preprocess categorical (same as in training)
    input_encoded = pd.get_dummies(input_df)

    # Align columns
    input_encoded = input_encoded.reindex(columns=model_input_columns, fill_value=0)

    # Scale input
    input_scaled = scaler.transform(input_encoded)

    # Predict
    pred = model.predict(input_scaled)
    prob = model.predict_proba(input_scaled)[0][1]

    return JSONResponse({
        "churn_prediction": int(pred[0]),
        "churn_probability": round(prob, 3)
    })

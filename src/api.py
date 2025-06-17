from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load model and artifacts
model = joblib.load("models/model.pkl")
encoder = joblib.load("models/encoder.pkl")
scaler = joblib.load("models/scaler.pkl")
columns = joblib.load("models/columns.pkl")
best_model_name = joblib.load("models/best_model_name.pkl")

app = FastAPI(title="Customer Churn Prediction API")

# Define input schema
class InputData(BaseModel):
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

@app.get("/")
def root():
    return {"message": "Churn Prediction API is running!"}

@app.post("/predict")
def predict(data: InputData):
    df_input = pd.DataFrame([data.dict()])

    # One-hot encode seperti di training
    df_input = pd.get_dummies(df_input)

    # Tambahkan kolom yang mungkin hilang
    df_input = df_input.reindex(columns=columns, fill_value=0)


    # Scale
    df_input = scaler.transform(df_input)

    # Predict
    prediction = model.predict(df_input)[0]
    result = "Yes" if prediction == 1 else "No"
    return {
        "prediction": result,
        "model_used": best_model_name
    }

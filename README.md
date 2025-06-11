## Customer Churn Prediction – MLOps Pipeline

This project implements a complete **MLOps pipeline** for customer churn prediction using structured telco data. It manages the full lifecycle of a machine learning model — from preprocessing and training to deployment and inference via API.

---

## Project Overview

| Stage                | Tool / Method          |
|----------------------|------------------------|
| Data Preprocessing   | pandas, scikit-learn   |
| Model Training       | RandomForestClassifier |
| Experiment Tracking  | MLflow                 |
| Model Deployment     | FastAPI                |
| API Testing          | Swagger UI             |

---

## Folder Structure

```
customer-churn-mlops/
├── data/               # Dataset CSV (not pushed to GitHub)
├── models/             # Saved model, scaler, and column metadata
├── src/
│   ├── preprocess.py   # Handles encoding, scaling
│   ├── train.py        # Trains model + logs to MLflow
│   └── api.py          # FastAPI for live prediction
├── requirements.txt    # Python dependencies
├── .gitignore          # Ignore venv, model files, logs, etc.
```

---

## Setup Instructions

### 1. Create and activate virtual environment
```bash
python -m venv venv
.\venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model
```bash
python -m src.train
```
- This will:
  - Preprocess the dataset
  - Train a model
  - Log the run to **MLflow**
  - Save: `model.pkl`, `scaler.pkl`, `columns.pkl`

### 4. (Optional) View MLflow experiment tracking
```bash
mlflow ui
```
Open in browser: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## Run the API Server

```bash
uvicorn src.api:app --reload
```

Visit Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)  
Use the `/predict` endpoint to send customer data and receive churn prediction.

---

## Sample Input (for `/predict`)

```json
{
  "gender": "Female",
  "SeniorCitizen": 1,
  "Partner": "No",
  "Dependents": "No",
  "tenure": 1,
  "PhoneService": "Yes",
  "MultipleLines": "Yes",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 95.0,
  "TotalCharges": 95.0
}
```

Expected result:
```json
{
  "churn_prediction": 1,
  "churn_probability": 0.92
}
```

---

## Notes

- Dataset (`customer_churn.csv`) is not pushed to GitHub; add manually to `/data/`
- Model files (`.pkl`) are generated after running `train.py`
- The pipeline can be extended with GitHub Actions and Docker for full CI/CD

---